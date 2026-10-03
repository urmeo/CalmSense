import { createElement, useSyncExternalStore } from 'react';
import createPlotlyComponent, { type PlotParams } from 'react-plotly.js/factory';
import Plotly from 'plotly.js/dist/plotly-basic.min.js';

const Plot = createPlotlyComponent(Plotly);
const isDark = () => document.documentElement.classList.contains('dark');
const subscribeTheme = (notify: () => void) => {
  const observer = new MutationObserver(notify);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  return () => observer.disconnect();
};

export default function ThemedPlot(props: PlotParams) {
  const dark = useSyncExternalStore(subscribeTheme, isDark, () => false);
  const layout = (props.layout ?? {}) as Record<string, unknown> & { font?: Record<string, unknown> };
  return createElement(Plot, {
    ...props,
    layout: { ...layout, font: { ...layout.font, color: dark ? '#D1D5DB' : '#4B5563' } },
  });
}
