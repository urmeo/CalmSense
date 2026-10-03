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
  const config = (props.config ?? {}) as Record<string, unknown>;
  return createElement(Plot, {
    ...props,
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      ...layout,
      font: { ...layout.font, color: dark ? '#D1D5DB' : '#4B5563' },
    },
    config: { responsive: true, showSendToCloud: false, displayModeBar: false, ...config },
    style: { width: '100%', ...props.style },
  });
}
