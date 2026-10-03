import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Activity, Brain, Layers, Award } from 'lucide-react';
import results from '../../../outputs/dashboard/results';
import Panel from '../components/Panel';
import { requiresFreshBenchmark } from '../lib/benchmarks';

const r = results as any;

// Static color classes (dynamic `bg-${color}-100` would be purged by Tailwind)
const CARD_COLORS: Record<string, string> = {
  blue: 'bg-blue-100 dark:bg-blue-900/30',
  green: 'bg-green-100 dark:bg-green-900/30',
  purple: 'bg-purple-100 dark:bg-purple-900/30',
  orange: 'bg-orange-100 dark:bg-orange-900/30',
};

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: string;
}> = ({ title, value, icon, color = 'blue' }) => (
  <Panel>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
        <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">{value}</p>
      </div>
      <div className={`p-3 rounded-lg ${CARD_COLORS[color]}`}>{icon}</div>
    </div>
  </Panel>
);

const pct = (x: number | null | undefined) => typeof x === 'number' && Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : 'Unavailable';

const FeatureImportanceChart: React.FC = () => {
  const palette = ['#3182CE', '#38A169', '#D69E2E', '#E53E3E', '#805AD5', '#DD6B20', '#319795', '#D53F8C'];
  const data = (r.shap || []).slice(0, 8).map((s: any, i: number) => ({
    name: s.feature,
    value: s.mean_abs_shap,
    color: palette[i % palette.length],
  }));

  return (
    <Panel title="Top features (binary XGBoost, full-data SHAP)">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis type="number" />
          {/* width fits the longest feature name (e.g. EDA_SCR_recovery_time_mean); a narrow
              axis clips the leftmost characters and makes real features look corrupted. */}
          <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={200} interval={0} />
          <Tooltip />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry: { color: string }, index: number) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  );
};

const OptimismGapChart: React.FC = () => {
  const b = r.binary || {};
  const loso = b.loso_matched_accuracy;
  const within = b.within_subject_accuracy;
  const hasMatched = typeof loso === 'number' && typeof within === 'number';
  const data = [
    { name: 'LOSO\n(subject-independent)', value: loso, color: '#3182CE' },
    { name: 'Subject-mixed\n(5-fold)', value: within, color: '#E67E22' },
  ];
  return (
    <Panel title="Optimism gap">
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        {hasMatched ? `Subject-mixed validation adds ${b.optimism_gap_pts} points on matched non-overlapping windows` : 'Matched-window comparison is unavailable for the selected benchmark model.'}
      </p>
      {hasMatched && <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip formatter={(v) => pct(Number(v))} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>}
    </Panel>
  );
};

const ModelComparisonList: React.FC = () => {
  const models = [...((r.binary || {}).models || [])].sort(
    (a: any, b: any) => b.accuracy_mean - a.accuracy_mean
  );
  return (
    <Panel title="Binary LOSO accuracy by model · subject means">
      <div className="space-y-3">
        {models.map((m: any) => (
          <div key={m.model} className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400 w-40">{m.model}</span>
            <div className="flex-1 mx-3 bg-gray-100 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${m.accuracy_mean * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white w-16 text-right">
              {pct(m.accuracy_mean)}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
};

const DatasetSummary: React.FC = () => {
  const b = r.binary || {};
  const m = r.multiclass || {};
  const rows = [
    ['Dataset', `WESAD (chest, ${b.n_subjects ?? 15} subjects)`],
    ['Windows (binary)', b.n_windows],
    ['Features', b.n_features],
    ['Binary classes', (b.classes || []).join(', ')],
    ['3-class classes', (m.classes || []).join(', ')],
    ['Validation', 'Leave-One-Subject-Out'],
  ];
  return (
    <Panel title="Dataset & setup">
      <div className="space-y-3">
        {rows.map(([k, v]) => (
          <div key={k as string} className="flex items-center justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">{k}</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">{v}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
};

const Dashboard: React.FC = () => {
  const b = r.binary || {};
  const m = r.multiclass || {};

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Subject-independent stress detection: real LOSO results
        </p>
      </div>

      {requiresFreshBenchmark(r) && (
        <p role="note" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-200">
          Saved benchmark snapshots use the earlier pipeline. The corrected code requires a fresh benchmark.
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Binary LOSO accuracy" value={pct(b.loso_accuracy)} icon={<Activity className="w-6 h-6 text-blue-600" />} color="blue" />
        <MetricCard title="3-class LOSO accuracy" value={pct(m.loso_accuracy)} icon={<Brain className="w-6 h-6 text-green-600" />} color="green" />
        <MetricCard title="Features" value={b.n_features} icon={<Layers className="w-6 h-6 text-purple-600" />} color="purple" />
        <MetricCard title="Highest observed accuracy" value={b.best_model} icon={<Award className="w-6 h-6 text-orange-600" />} color="orange" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeatureImportanceChart />
        <OptimismGapChart />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelComparisonList />
        <DatasetSummary />
      </div>
    </div>
  );
};

export default Dashboard;
