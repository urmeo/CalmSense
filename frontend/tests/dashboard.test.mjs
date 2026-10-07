import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readDarkMode, saveDarkMode } from '../src/lib/preferences.ts';
import { zoomRange, dataZoomRange } from '../src/lib/viewport.ts';
import { requiresFreshBenchmark, benchmarkStatus, formatPercent, matchedGap } from '../src/lib/benchmarks.ts';
import { readSignalRecording, recordingDuration, conditionSegments } from '../src/lib/signals.ts';
import { containNavigationFocus } from '../src/lib/navigation.ts';

test('corrupt theme preferences do not crash startup or enable dark mode', () => {
  for (const saved of [null, '', '{invalid', 'null', '1', 'false', '"true"']) {
    globalThis.window = { localStorage: { getItem: () => saved } };
    assert.equal(readDarkMode(), false);
  }
  globalThis.window = { localStorage: { getItem: () => 'true' } };
  assert.equal(readDarkMode(), true);
});

test('blocked storage does not prevent theme changes', () => {
  globalThis.window = {
    get localStorage() { throw new Error('Storage blocked'); },
  };
  assert.equal(readDarkMode(), false);
  assert.doesNotThrow(() => saveDarkMode(true));
  let saved;
  globalThis.window = { localStorage: { setItem: (key, value) => { saved = [key, value]; } } };
  saveDarkMode(true);
  assert.deepEqual(saved, ['darkMode', 'true']);
});

test('zoom out preserves its requested width at recording edges', () => {
  assert.deepEqual(zoomRange([0, 10], 120, 2), [0, 20]);
  assert.deepEqual(zoomRange([110, 120], 120, 2), [100, 120]);
  assert.deepEqual(zoomRange([0, 120], 120, 2), [0, 120]);
  assert.deepEqual(zoomRange([40, 80], 120, 0.5), [50, 70]);
});

test('historical exports stay marked until a corrected benchmark is recorded', () => {
  assert.equal(requiresFreshBenchmark({}), true);
  assert.equal(requiresFreshBenchmark({ benchmark_protocol_version: 1 }), true);
  assert.equal(requiresFreshBenchmark({ benchmark_protocol_version: 2 }), false);
  assert.equal(requiresFreshBenchmark({ benchmark_protocol_version: 3 }), false);
  for (const version of [NaN, Infinity, 2.1, -1]) {
    assert.equal(requiresFreshBenchmark({ benchmark_protocol_version: version }), true);
  }
});

test('primary protocol status does not certify unrelated ancillary snapshots', () => {
  assert.deepEqual(benchmarkStatus({}), {
    historical: true,
    primary: 'Primary benchmark snapshots use the earlier pipeline. The corrected code requires a fresh benchmark.',
    ancillary: null,
  });
  assert.deepEqual(benchmarkStatus({ benchmark_protocol_version: 2 }), {
    historical: false, primary: 'Primary benchmark: protocol v2.', ancillary: null,
  });
  const mixed = benchmarkStatus({
    benchmark_protocol_version: 2,
    unverified_sections: ['calibration', 'cross_dataset', 'wrist', 'shap', 'calibration'],
  });
  assert.equal(mixed.primary, 'Primary benchmark: protocol v2.');
  assert.equal(mixed.ancillary,
    'Additional snapshots lack verified linkage to this benchmark: calibration, transfer, wrist comparison, SHAP.');
  assert.equal(benchmarkStatus({ benchmark_protocol_version: 1, unverified_sections: ['calibration'] }).historical, true);
  const olderExport = benchmarkStatus({ benchmark_protocol_version: 2, calibration: {}, wrist: {}, shap: [], cross_dataset: null });
  assert.equal(olderExport.ancillary,
    'Additional snapshots lack verified linkage to this benchmark: SHAP, wrist comparison, calibration.');
  assert.equal(benchmarkStatus({ benchmark_protocol_version: 2, calibration: {}, unverified_sections: [] }).ancillary, null);
});

test('saved run summaries use recorded dates without inventing missing metadata', () => {
  const metadata = { benchmark_protocol_version: 2, binary: { n_subjects: 15 } };
  assert.equal(benchmarkStatus(metadata, '2026-10-04T09:05:02.671772+00:00').primary,
    'Saved results: 2026-10-04 · 15 subjects · LOSO.');
  for (const date of [undefined, '', 'today', '2026-02-30T00:00:00Z', '2026-13-01T00:00:00Z']) {
    assert.equal(benchmarkStatus(metadata, date).primary, 'Primary benchmark: protocol v2.');
  }
  assert.equal(benchmarkStatus({ benchmark_protocol_version: 2 }, '2026-10-04T00:00:00Z').primary,
    'Saved results: 2026-10-04.');
  const historical = benchmarkStatus({ ...metadata, benchmark_protocol_version: 1 }, '2026-10-04T00:00:00Z');
  assert.equal(historical.historical, true);
  assert.equal(historical.primary.includes('Saved results'), false);
});

test('missing or nonfinite comparison metrics remain unavailable rather than zero', () => {
  for (const value of [undefined, null, NaN, Infinity]) {
    assert.equal(formatPercent(value), 'Unavailable');
    assert.equal(matchedGap(value, 0.9), undefined);
    assert.equal(matchedGap(0.8, value), undefined);
  }
  assert.equal(formatPercent(0), '0.0%');
  assert.equal(formatPercent(0.9132857382783859), '91.3%');
  assert.ok(Math.abs(matchedGap(0.9, 0.85) + 5) < 1e-10);
});

test('chart zoom events reject malformed values and clamp panning to the clip', () => {
  for (const event of [null, 1, {}, { start: '0', end: 10 }, { start: 0 },
    { start: 0, end: Infinity }, { start: 10, end: 0 }, { start: 120, end: 150 },
    { batch: [] }, { batch: [null] }, { startValue: 0, endValue: NaN }]) {
    assert.equal(dataZoomRange(event, 120), null);
  }
  assert.deepEqual(dataZoomRange({ start: -10, end: 25 }, 120), [0, 30]);
  assert.deepEqual(dataZoomRange({ startValue: 100, endValue: 140 }, 120), [100, 120]);
  assert.deepEqual(dataZoomRange({ batch: [{ start: 0, end: 100 }] }, 120), [0, 120]);
  assert.deepEqual(dataZoomRange({ batch: [{ start: 25, end: 50 }] }, 120), [30, 60]);
  assert.equal(dataZoomRange({ start: 0, end: 100 }, 0), null);
  assert.equal(dataZoomRange({ start: 0, end: 100 }, NaN), null);
});

test('repeated zoom keeps at least one sample interval and recovers invalid ranges', () => {
  let range = [0, 120];
  for (let i = 0; i < 1000; i++) range = zoomRange(range, 120, 0.5, 1 / 30);
  assert.ok(range[1] - range[0] >= 1 / 30 - 1e-12);
  assert.deepEqual(zoomRange([NaN, 10], 120, 0.5), [0, 120]);
  assert.deepEqual(zoomRange([5, 5], 120, 0.5), [0, 120]);
  assert.deepEqual(zoomRange([0, 120], 120, 0), [0, 120]);
});

const clip = () => ({
  time: [0, 0.5, 1], ecg: [1, 2, 3], eda: [1, 2, 3], temp: [30, 31, 32],
  accX: [0, 0, 0], accY: [0, 0, 0], accZ: [1, 1, 1],
  conditions: ['Baseline', 'Stress', 'Stress'],
});

test('signal clips require aligned finite channels and increasing display times', () => {
  assert.equal(readSignalRecording(null), null);
  assert.equal(readSignalRecording({}), null);
  for (const changes of [{ time: [] }, { time: [0] }, { time: [0, 0, 1] },
    { time: [0, 1, 0.5] }, { time: [0, 0.5, NaN] }, { eda: [1, 2] },
    { accZ: [0, Infinity, 1] }, { ecg: [true, 2, 3] },
    { conditions: ['Baseline'] }, { conditions: [1, 'Stress', 'Stress'] }]) {
    assert.equal(readSignalRecording({ ...clip(), ...changes }), null);
  }
  const recording = readSignalRecording(clip());
  assert.equal(recordingDuration(recording), 1.5);
  assert.deepEqual(conditionSegments(recording), [
    { name: 'Baseline', x0: 0, x1: 0.5 }, { name: 'Stress', x0: 0.5, x1: 1.5 },
  ]);
});

test('the saved signal exports remain valid and all condition segments cover the clip', async () => {
  const { default: recordings } = await import('../../outputs/dashboard/signals.ts');
  assert.ok(Object.keys(recordings).length > 0);
  for (const row of Object.values(recordings)) {
    const recording = readSignalRecording(row);
    assert.ok(recording);
    const segments = conditionSegments(recording);
    assert.equal(segments[0].x0, 0);
    assert.equal(segments.at(-1).x1, recordingDuration(recording));
    for (let i = 1; i < segments.length; i++) assert.equal(segments[i].x0, segments[i - 1].x1);
  }
});

test('mobile menu contains keyboard focus, closes on Escape and restores its trigger', () => {
  const listeners = new Map();
  const document = {
    activeElement: null,
    addEventListener: (type, callback) => listeners.set(type, callback),
    removeEventListener: (type) => listeners.delete(type),
  };
  class Element {
    isConnected = true;
    ownerDocument = document;
    focus() { document.activeElement = this; listeners.get('focusin')?.(); }
  }
  const originalHTMLElement = globalThis.HTMLElement;
  globalThis.HTMLElement = Element;
  try {
    const trigger = new Element();
    const first = new Element();
    const last = new Element();
    const sidebar = new Element();
    sidebar.querySelectorAll = () => [first, last];
    sidebar.contains = (element) => element === first || element === last;
    const background = { inert: false };
    trigger.focus();
    let closed = 0;
    const release = containNavigationFocus(sidebar, background, () => closed++);
    assert.equal(background.inert, true);
    assert.equal(document.activeElement, first);
    let prevented = 0;
    listeners.get('keydown')({ key: 'Tab', shiftKey: true, preventDefault: () => prevented++ });
    assert.equal(document.activeElement, last);
    listeners.get('keydown')({ key: 'Tab', shiftKey: false, preventDefault: () => prevented++ });
    assert.equal(document.activeElement, first);
    trigger.focus();
    assert.equal(document.activeElement, first);
    listeners.get('keydown')({ key: 'Escape', preventDefault: () => prevented++ });
    assert.equal(closed, 1);
    assert.equal(prevented, 3);
    release();
    assert.equal(background.inert, false);
    assert.equal(document.activeElement, trigger);
    assert.equal(listeners.size, 0);
  } finally {
    globalThis.HTMLElement = originalHTMLElement;
  }
});
