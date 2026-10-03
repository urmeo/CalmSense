import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readDarkMode, saveDarkMode } from '../src/lib/preferences.ts';
import { zoomRange } from '../src/lib/viewport.ts';
import { requiresFreshBenchmark } from '../src/lib/benchmarks.ts';

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
});
