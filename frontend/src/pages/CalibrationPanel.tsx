import React from 'react';
import Plot from '../components/Plot';
import { Gauge, Target, AlertTriangle, TrendingDown, Info } from 'lucide-react';
import results from '../data';
import Panel from '../components/Panel';
import SummaryCard from '../components/SummaryCard';

const fmt = (v: number) => v.toFixed(3);
const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;

const TRANSPARENT = 'rgba(0,0,0,0)';
const COLORS = {
  loso: '#3182CE',
  recal: '#38A169',
  sigmoid: '#805AD5',
  diagonal: '#9CA3AF',
};

const markerTrace = (x: number[], y: number[], name: string, color: string) => ({
  x,
  y,
  type: 'scatter' as const,
  mode: 'lines+markers' as const,
  name,
  line: { color },
  marker: { color },
});

const CalibrationPanel: React.FC = () => {
  const cal = results.calibration;
  if (!cal) return <p role="status" className="text-gray-600 dark:text-gray-300">Calibration results are unavailable for this benchmark.</p>;

  const { loso, loso_matched, within_subject, recalibrated_isotonic, recalibrated_sigmoid, decision_curve } = cal;
  const dc = decision_curve;

  const evaluations = [
    { name: 'LOSO', summary: loso, color: COLORS.loso },
    { name: 'LOSO + isotonic', summary: recalibrated_isotonic, color: COLORS.recal },
    { name: 'LOSO + sigmoid', summary: recalibrated_sigmoid, color: COLORS.sigmoid },
  ];
  const rows = [
    { key: 'Subject-mixed · matched non-overlapping', s: within_subject },
    { key: 'LOSO · matched non-overlapping', s: loso_matched },
    { key: 'LOSO · all windows', s: loso },
    { key: 'LOSO + isotonic · all windows', s: recalibrated_isotonic },
    { key: 'LOSO + sigmoid · all windows', s: recalibrated_sigmoid },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Gauge className="w-6 h-6" /> Calibration
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          Binary Random Forest confidence on unseen subjects: full-window LOSO and
          training-subject recalibration ({cal.n_windows} windows, {cal.n_bins} bins).
          The optimism comparison uses a separate matched non-overlapping subset.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard icon={<Target className="w-6 h-6 text-blue-600" />} label="Full-window LOSO ECE" value={fmt(loso.ece)} />
        <SummaryCard
          icon={<AlertTriangle className="w-6 h-6 text-orange-500" />}
          label="Matched optimism gap (ECE)"
          value={signed(cal.calibration_optimism_gap_ece)}
        />
        <SummaryCard
          icon={<TrendingDown className="w-6 h-6 text-green-600" />}
          label="Full-window isotonic ECE reduction"
          value={signed(cal.recalibration_reduction_ece)}
        />
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-4 text-sm text-orange-800 dark:text-orange-200">
        On matched non-overlapping windows, subject-mixed ECE is{' '}
        <strong>{fmt(within_subject.ece)}</strong> and LOSO ECE is{' '}
        <strong>{fmt(loso_matched.ece)}</strong>: a gap of{' '}
        <strong>{signed(cal.calibration_optimism_gap_ece)}</strong>.
        Separately, full-window LOSO ECE falls from <strong>{fmt(loso.ece)}</strong> to{' '}
        <strong>{fmt(recalibrated_isotonic.ece)}</strong> with training-subject isotonic recalibration.
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Reliability · all windows">
          <Plot
            data={[
              {
                x: [0, 1],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                name: 'Perfectly calibrated',
                line: { color: COLORS.diagonal, dash: 'dot' },
              },
              ...evaluations.map(({ name, summary, color }) => markerTrace(
                summary.reliability.map((r) => r.confidence),
                summary.reliability.map((r) => r.accuracy),
                `${name} (ECE ${fmt(summary.ece)})`,
                color
              )),
            ]}
            layout={{
              height: 380,
              margin: { l: 50, r: 20, t: 10, b: 50 },
              xaxis: { title: { text: 'Confidence' }, range: [0, 1] },
              yaxis: { title: { text: 'Accuracy' }, range: [0, 1] },
              legend: { x: 0.02, y: 0.98, bgcolor: TRANSPARENT, font: { size: 10 } },
            }}
          />
        </Panel>

        <Panel title="Expected calibration error · all windows">
          <Plot
            data={[
              {
                x: evaluations.map((e) => e.name),
                y: evaluations.map((e) => e.summary.ece),
                type: 'bar',
                marker: { color: evaluations.map((e) => e.color) },
                text: evaluations.map((e) => fmt(e.summary.ece)),
                textposition: 'outside',
                hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
              },
            ]}
            layout={{
              height: 380,
              margin: { l: 50, r: 20, t: 20, b: 50 },
              yaxis: { title: { text: 'ECE' }, rangemode: 'tozero' },
            }}
          />
        </Panel>
      </div>

      <Panel title="Decision-curve analysis">
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Exploratory net benefit on full-window LOSO predictions, versus alerting everyone
          or no one. No clinical deployment has been validated.
        </p>
        <Plot
          data={[
            markerTrace(dc.thresholds, dc.net_benefit_uncalibrated, 'Uncalibrated', COLORS.loso),
            markerTrace(dc.thresholds, dc.net_benefit_recalibrated, 'Recalibrated', COLORS.recal),
            {
              x: dc.thresholds,
              y: dc.treat_all,
              type: 'scatter',
              mode: 'lines',
              name: 'Alert everyone',
              line: { color: COLORS.diagonal, dash: 'dash' },
            },
            {
              x: dc.thresholds,
              y: dc.thresholds.map(() => 0),
              type: 'scatter',
              mode: 'lines',
              name: 'Alert no one',
              line: { color: '#6B7280', dash: 'dot' },
            },
          ]}
          layout={{
            height: 360,
            margin: { l: 60, r: 20, t: 10, b: 50 },
            xaxis: { title: { text: 'Alert threshold' } },
            yaxis: { title: { text: 'Net benefit' } },
            legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
          }}
        />
      </Panel>

      <Panel title="Calibration metrics by evaluation">
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Brier is the mean squared error of the stress probability, averaged over windows.
          Matched and full-window rows use different evaluation sets.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-300">
                <th className="px-4 py-2">Evaluation</th>
                <th className="px-4 py-2">ECE</th>
                <th className="px-4 py-2">MCE</th>
                <th className="px-4 py-2">Brier</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key} className="border-b border-gray-100 dark:border-gray-700">
                  <td className="px-4 py-2 font-medium text-gray-900 dark:text-white">{row.key}</td>
                  <td className="px-4 py-2">{fmt(row.s.ece)}</td>
                  <td className="px-4 py-2">{fmt(row.s.mce)}</td>
                  <td className="px-4 py-2">{fmt(row.s.brier)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="flex items-start gap-2 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm">
        <Info className="w-5 h-5 text-blue-500 mt-0.5 shrink-0" />
        <p className="text-blue-800 dark:text-blue-300">
          Recalibration fits out-of-fold probabilities from the training subjects; the held-out
          subject is excluded. Research use only; no clinical validation.
        </p>
      </div>
    </div>
  );
};

export default CalibrationPanel;
