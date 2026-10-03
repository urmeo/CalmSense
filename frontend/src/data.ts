import savedResults from '../../outputs/dashboard/results';
import type { BenchmarkResults } from './types';

// The exporter validates required results; optional experiments may be absent.
const results: BenchmarkResults = savedResults;
export default results;
