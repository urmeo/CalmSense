// Results types (the dashboard renders precomputed JSON; there is no backend)

// Calibration analysis (results.calibration, written by scripts/calibration.py)

export interface ReliabilityBin {
  confidence: number;
  accuracy: number;
  count: number;
}

export interface CalibrationSummary {
  ece: number;
  mce: number;
  brier: number;
  reliability: ReliabilityBin[];
}

export interface DecisionCurve {
  thresholds: number[];
  net_benefit_uncalibrated: number[];
  net_benefit_recalibrated: number[];
  treat_all: number[];
}

export interface Calibration {
  model: string;
  positive_class: string;
  n_windows: number;
  n_bins: number;
  loso: CalibrationSummary;
  within_subject: CalibrationSummary;
  recalibrated_isotonic: CalibrationSummary;
  recalibrated_sigmoid: CalibrationSummary;
  calibration_optimism_gap_ece: number;
  recalibration_reduction_ece: number;
  decision_curve: DecisionCurve;
}
