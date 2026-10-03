export interface BenchmarkMetadata {
  benchmark_protocol_version?: number;
}

export function requiresFreshBenchmark(metadata: BenchmarkMetadata): boolean {
  return (metadata.benchmark_protocol_version ?? 1) < 2;
}
