import React, { useState } from 'react';
import { Trophy, AlertTriangle, Activity } from 'lucide-react';
import results from '../data';
import Panel from '../components/Panel';
import Chart from '../components/Chart';
import { barChartOption } from '../lib/charts';
import SummaryCard from '../components/SummaryCard';
import { formatPercent as pct, matchedGap } from '../lib/benchmarks';

type Task = 'binary' | 'multiclass';

const ModelComparison: React.FC = () => {
  const [task, setTask] = useState<Task>('binary');
  const data = results[task];
  const models = [...data.models].sort(
    (a, b) => b.accuracy_mean - a.accuracy_mean
  );
  const best = models[0];
  const losoMatched = data.loso_matched_accuracy;
  const gap = matchedGap(losoMatched, data.within_subject_accuracy);
  const wrist = results.wrist?.same_model_rf;
  const transfer = results.cross_dataset;
  const shap = results.shap || [];

  const barData = models.map((m) => ({
    name: m.model,
    accuracy: +(m.accuracy_mean * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Comparison</h1>
          <p className="text-gray-500 dark:text-gray-400">
            WESAD: {data.n_subjects ?? 15} held-out subjects. Table accuracy and macro-F1 are subject means;
            balanced accuracy pools held-out predictions.
          </p>
        </div>
        <select
          aria-label="Classification task"
          value={task}
          onChange={(e) => {
            const value = e.target.value;
            if (value === 'binary' || value === 'multiclass') setTask(value);
          }}
          className="max-w-full shrink-0 px-3 py-1.5 bg-gray-100 dark:bg-gray-700 border-0 rounded-lg text-sm"
        >
          <option value="binary">Binary (baseline vs. stress)</option>
          <option value="multiclass">3-class (baseline/stress/amusement)</option>
        </select>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard icon={<Trophy className="w-6 h-6 text-green-600" />} label="Highest observed accuracy" value={best?.model ?? 'Unavailable'} />
        <SummaryCard
          icon={<Activity className="w-6 h-6 text-blue-600" />}
          label="LOSO accuracy"
          value={pct(best?.accuracy_mean)}
        />
        <SummaryCard
          icon={<AlertTriangle className="w-6 h-6 text-orange-500" />}
          label="Matched optimism gap"
          value={gap === undefined ? 'Unavailable' : `${gap >= 0 ? '+' : ''}${gap.toFixed(1)} pts`}
        />
      </div>

      {gap !== undefined && <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-4 text-sm text-orange-800 dark:text-orange-200">
        On the same non-overlapping windows, {data.best_model} pooled accuracy is{' '}
        <strong>{pct(data.within_subject_accuracy)}</strong> under subject-mixed 5-fold validation
        versus <strong>{pct(losoMatched)}</strong> under LOSO. The gap is{' '}
        <strong>{gap.toFixed(1)} percentage points</strong>.
      </div>}

      {task === 'binary' && (wrist || transfer) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {wrist && <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">Binary RF: wrist vs. chest</p>
            <p className="text-gray-600 dark:text-gray-400">
              With the same model, Empatica E4 wrist signals reach{' '}
              <strong>{pct(wrist.wrist)}</strong> vs{' '}
              {pct(wrist.chest)} for the chest. Chest-to-wrist drop:{' '}
              {typeof wrist.drop_pts === 'number' && Number.isFinite(wrist.drop_pts) ? `${wrist.drop_pts.toFixed(1)} pts.` : 'unavailable.'}{' '}
              This laboratory comparison does not establish sensor equivalence or field performance.
            </p>
          </div>}
          {transfer && <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 text-sm">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">Binary cross-dataset transfer</p>
            <p className="text-gray-600 dark:text-gray-400">
              Balanced accuracy: WESAD → PhysioNet Non-EEG{' '}
              <strong>{pct(transfer.wesad_to_noneeg?.balanced_accuracy)}</strong>;
              reverse transfer{' '}
              <strong>{pct(transfer.noneeg_to_wesad?.balanced_accuracy)}</strong>.
              Signal units, devices, stressors, and labels differ; this pair does not isolate dataset shift.
            </p>
          </div>}
        </div>
      )}

      <Panel title={`Subject-independent performance (${data.n_windows} windows)`}>
        {models.length === 0 && <p role="status" className="text-gray-500 dark:text-gray-400 mb-3">No model results are available for this task.</p>}
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
              {models.map((m, i) => (
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
      </Panel>

      <div className={`grid grid-cols-1 ${task === 'binary' ? 'lg:grid-cols-2' : ''} gap-6`}>
        <Panel title="LOSO accuracy">
          <Chart label={`${task} LOSO accuracy by model, from highest to lowest.`}
            option={barChartOption({ labels: barData.map((d) => d.name), values: barData.map((d) => d.accuracy),
              colors: barData.map((_, i) => i === 0 ? '#38A169' : '#3182CE'),
              axisName: 'Accuracy', maximum: 100, percent: true, precision: 1 })}
          />
        </Panel>

        {task === 'binary' && (
          <Panel title="Binary XGBoost · full-data SHAP">
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              Descriptive mean |SHAP| on the training data; not held-out or causal evidence.
            </p>
            {shap.length > 0 ? (
              <Chart label="Binary XGBoost full-data mean absolute SHAP contributions."
                option={barChartOption({ labels: shap.map((d) => d.feature), values: shap.map((d) => d.mean_abs_shap),
                  colors: ['#805AD5'], horizontal: true, axisName: 'mean |SHAP|' })}
              />
            ) : (
              <p className="text-sm text-gray-500">Run the experiment to populate SHAP values.</p>
            )}
          </Panel>
        )}
      </div>
    </div>
  );
};

export default ModelComparison;
