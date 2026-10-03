export function zoomRange(
  range: [number, number], duration: number, factor: number, minimumWidth = 0,
): [number, number] {
  if (!Number.isFinite(duration) || duration <= 0) return [0, 0];
  if (!range.every(Number.isFinite) || range[1] <= range[0] || !Number.isFinite(factor) || factor <= 0) return [0, duration];
  const width = Math.min(Math.max((range[1] - range[0]) * factor, minimumWidth, duration * 1e-9), duration);
  const center = (range[0] + range[1]) / 2;
  const start = Math.max(0, Math.min(center - width / 2, duration - width));
  return [start, start + width];
}

export function relayoutRange(event: unknown, duration: number): [number, number] | null {
  if (!event || typeof event !== 'object' || !Number.isFinite(duration) || duration <= 0) return null;
  const change = event as Record<string, unknown>;
  if (change['xaxis.autorange'] === true) return [0, duration];
  const range = change['xaxis.range'] ?? [change['xaxis.range[0]'], change['xaxis.range[1]']];
  if (!Array.isArray(range) || range.length !== 2
    || !range.every((value) => typeof value === 'number' && Number.isFinite(value))) return null;
  const start = Math.max(0, range[0]);
  const end = Math.min(duration, range[1]);
  return end > start ? [start, end] : null;
}
