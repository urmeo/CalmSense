const SECTION_LABELS = {
  shap: 'SHAP', stats: 'statistics', wrist: 'wrist comparison',
  cross_dataset: 'transfer', calibration: 'calibration',
  personalization: 'personalization', tuning: 'tuning', ablation: 'ablation',
} as const;

export interface BenchmarkMetadata extends Partial<Record<keyof typeof SECTION_LABELS, unknown>> {
  benchmark_protocol_version?: number;
  unverified_sections?: string[];
  binary?: { n_subjects?: number };
}

export function requiresFreshBenchmark(metadata: BenchmarkMetadata): boolean {
  const version = metadata.benchmark_protocol_version;
  return !Number.isInteger(version) || (version ?? 1) < 2;
}

export function benchmarkStatus(metadata: BenchmarkMetadata, runDate?: string) {
  const historical = requiresFreshBenchmark(metadata);
  const day = runDate?.match(/^(\d{4}-\d{2}-\d{2})T/)?.[1];
  const recordedDay = day && Number.isFinite(Date.parse(runDate!))
    && new Date(`${day}T00:00:00Z`).toISOString().startsWith(day) ? day : null;
  const subjects = metadata.binary?.n_subjects;
  const cohort = Number.isInteger(subjects) && (subjects ?? 0) >= 2
    ? ` · ${subjects} subjects · LOSO` : '';
  const labels: Record<string, string> = SECTION_LABELS;
  const present = Object.entries(SECTION_LABELS)
    .filter(([name]) => metadata[name as keyof typeof SECTION_LABELS] != null)
    .map(([name]) => name);
  // Primary protocol alone cannot certify ancillary runs.
  const sections = [...new Set(metadata.unverified_sections ?? (historical ? [] : present))];
  return {
    historical,
    primary: historical
      ? 'Primary benchmark snapshots use the earlier pipeline. The corrected code requires a fresh benchmark.'
      : recordedDay ? `Saved results: ${recordedDay}${cohort}.`
      : `Primary benchmark: protocol v${metadata.benchmark_protocol_version}.`,
    ancillary: sections.length > 0
      ? `Additional snapshots lack verified linkage to this benchmark: ${sections.map((name) => labels[name] ?? name).join(', ')}.`
      : null,
  };
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
