export function zoomRange(
  range: [number, number], duration: number, factor: number,
): [number, number] {
  const width = Math.min((range[1] - range[0]) * factor, duration);
  const center = (range[0] + range[1]) / 2;
  const start = Math.max(0, Math.min(center - width / 2, duration - width));
  return [start, start + width];
}
