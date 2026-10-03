export interface BenchmarkMetadata {
  benchmark_protocol_version?: number;
}

export function requiresFreshBenchmark(metadata: BenchmarkMetadata): boolean {
  const version = metadata.benchmark_protocol_version;
  return !Number.isInteger(version) || (version ?? 1) < 2;
}

export function formatPercent(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%` : 'Unavailable';
}

export function matchedGap(
  loso: number | null | undefined, withinSubject: number | null | undefined,
): number | undefined {
  return typeof loso === 'number' && Number.isFinite(loso)
    && typeof withinSubject === 'number' && Number.isFinite(withinSubject)
    ? (withinSubject - loso) * 100 : undefined;
}
