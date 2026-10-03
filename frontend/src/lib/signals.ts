export interface SignalRecording {
  time: number[];
  ecg: number[];
  eda: number[];
  temp: number[];
  accX: number[];
  accY: number[];
  accZ: number[];
  conditions: string[];
}

const channels = ['time', 'ecg', 'eda', 'temp', 'accX', 'accY', 'accZ'] as const;

export function readSignalRecording(value: unknown): SignalRecording | null {
  if (!value || typeof value !== 'object') return null;
  const row = value as Record<string, unknown>;
  const time = row.time;
  if (!Array.isArray(time) || time.length < 2 || time[0] !== 0) return null;
  for (const channel of channels) {
    const samples = row[channel];
    if (!Array.isArray(samples) || samples.length !== time.length
      || !samples.every((sample) => typeof sample === 'number' && Number.isFinite(sample))) return null;
  }
  if (!time.every((sample, i) => i === 0 || sample > time[i - 1])) return null;
  if (!Array.isArray(row.conditions) || row.conditions.length !== time.length
    || !row.conditions.every((condition) => typeof condition === 'string')) return null;
  return row as unknown as SignalRecording;
}

export function recordingDuration({ time }: SignalRecording): number {
  return time[time.length - 1] + (time[1] - time[0]);
}

export function conditionSegments(recording: SignalRecording) {
  const { conditions, time } = recording;
  const segments: { name: string; x0: number; x1: number }[] = [];
  let start = 0;
  for (let i = 1; i <= conditions.length; i++) {
    if (i === conditions.length || conditions[i] !== conditions[start]) {
      segments.push({ name: conditions[start], x0: time[start], x1: i < time.length ? time[i] : recordingDuration(recording) });
      start = i;
    }
  }
  return segments;
}
