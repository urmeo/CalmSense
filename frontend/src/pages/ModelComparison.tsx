import React, { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Trophy, AlertTriangle, Activity } from 'lucide-react';
import results from '../../../outputs/dashboard/results';
import SummaryCard from '../components/SummaryCard';

type Task = 'binary' | 'multiclass';

const pct = (v: number | null | undefined) => typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : 'Unavailable';

const ModelComparison: React.FC = () => {
  const [task, setTask] = useState<Task>('binary');
  const data = (results as any)[task];
  const models = [...data.models].sort(
    (a: any, b: any) => b.accuracy_mean - a.accuracy_mean
  );
  const best = models[0];
  const losoMatched = data.loso_matched_accuracy;
  const gap = typeof data.within_subject_accuracy === 'number' && typeof losoMatched === 'number'
    ? (data.within_subject_accuracy - losoMatched) * 100 : undefined;
  const shap = (results as any).shap || [];

  const barData = models.map((m: any) => ({
    name: m.model,
    accuracy: +(m.accuracy_mean * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Comparison</h1>
          <p className="text-gray-500 dark:text-gray-400">
            WESAD: {data.n_subjects ?? 15} held-out subjects. Accuracy and macro-F1 are subject means;
            balanced accuracy pools held-out predictions.
          </p>
        </div>
        <select
          aria-label="Classification task"
          value={task}
          onChange={(e) => setTask(e.target.value as Task)}
          className="px-3 py-1.5 bg-gray-100 dark:bg-gray-700 border-0 rounded-lg text-sm"
        >
          <option value="binary">Binary (baseline vs. stress)</option>
          <option value="multiclass">3-class (baseline/stress/amusement)</option>
        </select>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard icon={<Trophy className="w-6 h-6 text-green-600" />} label="Highest observed accuracy" value={best.model} />
        <SummaryCard
          icon={<Activity className="w-6 h-6 text-blue-600" />}
          label="LOSO accuracy"
          value={pct(best.accuracy_mean)}
        />
        <SummaryCard
          icon={<AlertTriangle className="w-6 h-6 text-orange-500" />}
          label="Matched optimism gap"
          value={gap === undefined ? 'Unavailable' : `${gap >= 0 ? '+' : ''}${gap.toFixed(1)} pts`}
        />
      </div>

      {gap !== undefined && <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-4 text-sm text-orange-800 dark:text-orange-200">
        On the same non-overlapping windows, {data.best_model} scores{' '}
        <strong>{pct(data.within_subject_accuracy)}</strong> under subject-mixed 5-fold validation
        versus <strong>{pct(losoMatched)}</strong> under LOSO. The gap is{' '}
        <strong>{gap.toFixed(1)} percentage points</strong>.
      </div>}

      {task === 'binary' && (results as any).cross_dataset && (results as any).wrist && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">Binary RF: wrist vs. chest</p>
            <p className="text-gray-600 dark:text-gray-400">
              With the same model, Empatica E4 wrist signals reach{' '}
              <strong>{pct((results as any).wrist.same_model_rf.wrist)}</strong> vs{' '}
              {pct((results as any).wrist.same_model_rf.chest)} for the chest, a{' '}
              {(results as any).wrist.same_model_rf.drop_pts.toFixed(1)}-pt drop.
              This laboratory comparison does not establish sensor equivalence or field performance.
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">Binary cross-dataset transfer</p>
            <p className="text-gray-600 dark:text-gray-400">
              Balanced accuracy: WESAD → PhysioNet Non-EEG{' '}
              <strong>{pct((results as any).cross_dataset.wesad_to_noneeg.balanced_accuracy)}</strong>;
              reverse transfer{' '}
              <strong>{pct((results as any).cross_dataset.noneeg_to_wesad.balanced_accuracy)}</strong>.
              Devices, stressors, and labels differ; this pair does not isolate dataset shift.
            </p>
          </div>
        </div>
      )}

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Subject-independent performance ({data.n_windows} windows)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-300">
                <th className="px-4 py-2">Model</th>
                <th className="px-4 py-2">Accuracy</th>
                <th className="px-4 py-2">Macro-F1</th>
                <th className="px-4 py-2">Balanced acc.</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m: any, i: number) => (
                <tr
                  key={m.model}
                  className={`border-b border-gray-100 dark:border-gray-700 ${
                    i === 0 ? 'bg-green-50 dark:bg-green-900/20' : ''
                  }`}
                >
                  <td className="px-4 py-2 font-medium text-gray-900 dark:text-white flex items-center gap-2">
                    {i === 0 && <Trophy className="w-4 h-4 text-yellow-500" />}
                    {m.model}
                  </td>
                  <td className="px-4 py-2">{pct(m.accuracy_mean)}</td>
                  <td className="px-4 py-2">{pct(m.f1_macro_mean)}</td>
                  <td className="px-4 py-2">{pct(m.balanced_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className={`grid grid-cols-1 ${task === 'binary' ? 'lg:grid-cols-2' : ''} gap-6`}>
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">LOSO accuracy</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="name" angle={-30} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} unit="%" />
              <Tooltip formatter={(v) => `${v}%`} />
              <Legend />
              <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                {barData.map((_: any, i: number) => (
                  <Cell key={i} fill={i === 0 ? '#38A169' : '#3182CE'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {task === 'binary' && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Binary XGBoost · full-data SHAP
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              Descriptive mean |SHAP| on the training data; not held-out or causal evidence.
            </p>
            {shap.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={shap} layout="vertical" margin={{ left: 40, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="feature" width={175} tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="mean_abs_shap" name="mean |SHAP|" fill="#805AD5" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-gray-500">Run the experiment to populate SHAP values.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelComparison;
