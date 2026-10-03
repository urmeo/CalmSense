interface ReliabilityBin {
  confidence: number;
  accuracy: number;
}

interface CalibrationSummary {
  ece: number;
  mce: number;
  brier: number;
  reliability: ReliabilityBin[];
}

interface DecisionCurve {
  thresholds: number[];
  net_benefit_uncalibrated: number[];
  net_benefit_recalibrated: number[];
  treat_all: number[];
}

export interface Calibration {
  n_windows: number;
  n_bins: number;
  loso: CalibrationSummary;
  loso_matched: CalibrationSummary;
  within_subject: CalibrationSummary;
  recalibrated_isotonic: CalibrationSummary;
  recalibrated_sigmoid: CalibrationSummary;
  calibration_optimism_gap_ece: number;
  recalibration_reduction_ece: number;
  decision_curve: DecisionCurve;
}
