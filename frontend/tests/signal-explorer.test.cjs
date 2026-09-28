const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const React = require('react');
const ts = require('typescript');

const signal = {
  time: [0, 30, 60, 90, 120],
  ecg: [1, 2, 3, 2, 1],
  eda: [2, 3, 4, 3, 2],
  temp: [30, 31, 32, 31, 30],
  accX: [0, 1, 0, -1, 0],
  accY: [1, 0, -1, 0, 1],
  accZ: [0, -1, 0, 1, 0],
  conditions: ['Baseline', 'Baseline', 'Stress', 'Stress', 'Amusement'],
};
const groups = [
  ['ecg', [['ecg', 'ECG', '#E53E3E']]],
  ['eda', [['eda', 'EDA', '#3182CE']]],
  ['temp', [['temp', 'Temperature', '#38A169']]],
  ['acc', [['accX', 'ACC X', '#805AD5'], ['accY', 'ACC Y', '#DD6B20'], ['accZ', 'ACC Z', '#319795']]],
];
const source = readFileSync(path.join(__dirname, '../src/pages/SignalExplorer.tsx'), 'utf8');
const code = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
}).outputText;

function render(visibleSignals, range = [0, 120]) {
  const state = ['synthetic', visibleSignals, range];
  let cursor = 0;
  const Plot = () => null;
  const module = { exports: {} };
  const mockedReact = {
    ...React,
    useState: () => {
      const index = cursor++;
      return [state[index], (next) => {
        state[index] = typeof next === 'function' ? next(state[index]) : next;
      }];
    },
    useMemo: (compute) => compute(),
    useEffect: () => {},
  };
  const load = (id) => {
    if (id === 'react') return mockedReact;
    if (id === '../components/Plot') return Plot;
    if (id === '../data/signals.json') return { synthetic: signal };
    return require(id);
  };
  vm.runInNewContext(code, { module, exports: module.exports, require: load });
  return { tree: module.exports.default(), state, Plot };
}

function find(element, predicate) {
  if (!element || typeof element !== 'object') return undefined;
  if (Array.isArray(element)) return element.map((child) => find(child, predicate)).find(Boolean);
  if (predicate(element)) return element;
  return find(element.props?.children, predicate);
}

test('all 16 signal combinations preserve samples, colors, order and stacked axes', () => {
  for (let mask = 0; mask < 16; mask++) {
    const visible = Object.fromEntries(groups.map(([key], index) => [key, Boolean(mask & (1 << index))]));
    const { tree, Plot } = render(visible);
    const traces = find(tree, (element) => element.type === Plot).props.data;
    const expected = [];
    let axis = 0;
    for (const [group, channels] of groups) {
      if (!visible[group]) continue;
      for (const [key, name, color] of channels) {
        expected.push({
          x: signal.time,
          y: signal[key],
          type: 'scatter',
          mode: 'lines',
          name,
          line: { color, width: 1 },
          xaxis: 'x',
          yaxis: axis === 0 ? 'y' : `y${axis + 1}`,
        });
      }
      axis++;
    }
    assert.deepEqual(JSON.parse(JSON.stringify(traces)), expected, `combination ${mask}`);
  }
});

for (const [name, range, expected] of [
  ['inside data', [40, 60], [30, 70]],
  ['across the start', [-20, 20], [0, 80]],
  ['across the end', [100, 140], [40, 120]],
  ['before data', [-100, -50], [0, 100]],
  ['after data', [170, 220], [20, 120]],
  ['full extent', [0, 120], [0, 120]],
]) {
  test(`zoom out keeps its full span within the data: ${name}`, () => {
    const { tree, state } = render({ ecg: true, eda: true, temp: true, acc: false }, range);
    find(tree, (element) => element.props?.title === 'Zoom Out').props.onClick();
    assert.deepEqual(Array.from(state[2]), expected);
  });
}
