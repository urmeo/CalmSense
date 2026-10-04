import { useEffect, useRef, useSyncExternalStore, type CSSProperties } from 'react';
import { init, use, type ComposeOption, type EChartsType } from 'echarts/core';
import { BarChart, LineChart, type BarSeriesOption, type LineSeriesOption } from 'echarts/charts';
import {
  AriaComponent, DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, TitleComponent,
  ToolboxComponent, TooltipComponent, type AriaComponentOption, type DataZoomComponentOption,
  type GridComponentOption, type LegendComponentOption, type TitleComponentOption,
  type ToolboxComponentOption, type TooltipComponentOption,
} from 'echarts/components';
import { SVGRenderer } from 'echarts/renderers';

use([BarChart, LineChart, AriaComponent, DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent,
  TitleComponent, ToolboxComponent, TooltipComponent, SVGRenderer]);

export type ChartOption = ComposeOption<BarSeriesOption | LineSeriesOption | AriaComponentOption
  | DataZoomComponentOption | GridComponentOption | LegendComponentOption | TitleComponentOption
  | ToolboxComponentOption | TooltipComponentOption>;

const isDark = () => document.documentElement.classList.contains('dark');
const subscribeTheme = (notify: () => void) => {
  const observer = new MutationObserver(notify);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  return () => observer.disconnect();
};

export default function Chart({ option, label, height = 300, onDataZoom, style }: {
  option: ChartOption;
  label: string;
  height?: number;
  onDataZoom?: (event: unknown) => void;
  style?: CSSProperties;
}) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<EChartsType | null>(null);
  const zoomHandler = useRef(onDataZoom);
  zoomHandler.current = onDataZoom;
  const dark = useSyncExternalStore(subscribeTheme, isDark, () => false);

  useEffect(() => {
    if (!element.current) return;
    const instance = init(element.current, dark ? 'dark' : undefined, { renderer: 'svg' });
    chart.current = instance;
    instance.on('datazoom', (event: unknown) => zoomHandler.current?.(event));
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
  }, [dark]);

  useEffect(() => {
    chart.current?.setOption({
      animation: false,
      backgroundColor: 'transparent',
      textStyle: { fontFamily: 'inherit' },
      aria: { enabled: true, label: { description: label } },
      tooltip: { trigger: 'axis', confine: true, renderMode: 'richText' },
      ...option,
    }, { replaceMerge: ['series', 'xAxis', 'yAxis', 'grid', 'dataZoom'] });
  }, [option, label, dark]);

  return <div ref={element} role="img" aria-label={label} style={{ width: '100%', height, ...style }} />;
}
