# CalmSense

### A Machine Learning and Deep Learning Framework for Wearable Biosignal Analysis, Integrating 1D CNNs, Explainable AI, Probability Calibration, and Cross-Dataset Evaluation.

ML: Logistic Regression, Random Forest, XGBoost, LightGBM

DL: 1D-CNN, SHAP

[Live demo](https://urmeo.github.io/CalmSense/) · [Colab](https://colab.research.google.com/github/urmeo/CalmSense/blob/main/notebooks/CalmSense.ipynb)

[![CalmSense dashboard](docs/demo.gif)](https://urmeo.github.io/CalmSense/)

## What this is

- Detects stress vs baseline from wearable signals: ECG, EDA (skin conductance), temperature, respiration, motion.
- Scored Leave-One-Subject-Out (LOSO): train on 14 people, test on the 15th, rotate.
- Shows where the usual high numbers come from: subject leakage, motion, dataset shift, calibration.
- Ships a static dashboard of the committed results (no backend). make demo runs the full pipeline offline on synthetic signals.

## Data and evaluation protocol

WESAD supplies 15 participants with chest RespiBAN signals at 700 Hz and wrist Empatica E4 signals
(BVP 64 Hz, EDA/temperature 4 Hz, accelerometer 32 Hz). Binary classification compares baseline
with stress; the three-class task adds amusement. Meditation is excluded as a recovery state.
PhysioNet Non-EEG supplies 20 participants for the separate transfer check. Dataset sources,
access terms, and checksums are in [README.md](README.md) and
[README.md](README.md); raw datasets are not redistributed.

Signals are filtered, ECG R-peaks receive ectopic correction, and EDA is separated into tonic and
phasic components. Windows last 60 seconds, overlap by 50%, and require at least 90% of samples
to share one retained label. The committed benchmark has 869 binary and 1,032 three-class windows.

All benchmark folds hold out an entire subject. Median imputation, standardization, and class
balancing are fit on training subjects only. The 1D-CNN operates on raw signal windows and uses
its own training and normalization path. Leakage comparisons use matched non-overlapping windows;
recalibration uses out-of-fold predictions from training subjects; personalization reserves a
separate half of the held-out subject's non-overlapping windows for evaluation.

### Models

| Model | Type | Key settings |
| --- | --- | --- |
| Logistic Regression | Linear | C=1.0, L2, class-balanced |
| Random Forest | Bagged trees | 200 trees, depth 10, class-balanced |
| XGBoost | Boosted trees | 200 trees, depth 7, learning rate 0.1 |
| LightGBM | Boosted trees | 200 trees, 50 leaves, learning rate 0.1 |
| 1D-CNN | Deep net on raw signal | Residual blocks, AdamW, early stopping |

*The four feature models share the impute/scale/classifier pipeline; the CNN is a small baseline.*

### Features (58)

| Group | Count | Examples |
| --- | ---: | --- |
| HRV time domain | 12 | MeanNN, SDNN, RMSSD, pNN50 |
| HRV frequency | 8 | LF/HF power, LF/HF ratio |
| HRV nonlinear | 10 | SampEn, DFA, SD1/SD2, CSI |
| EDA (skin conductance) | 15 | SCL level, SCR count, SCR amplitude |
| Temperature + respiration | 8 | Temperature slope, respiration rate |
| Accelerometer (motion) | 5 | Magnitude mean, standard deviation, energy |

*Feature extraction summarizes cardiac, autonomic, respiratory, thermal, and movement signals per window.*

## Benchmark results

### Binary classification

| Model | Accuracy | F1 (macro) | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.913 | 0.898 | 0.973 | 0.960 |
| XGBoost | 0.903 | 0.873 | 0.975 | 0.960 |
| Logistic Regression | 0.902 | 0.883 | 0.959 | 0.947 |
| LightGBM | 0.894 | 0.860 | 0.965 | 0.946 |
| 1D-CNN (raw signal) | 0.718 | 0.648 | n/a | n/a |

*Accuracy and macro-F1 are means over the 15 held-out subjects from
[metrics.json](results/metrics.json). AUROC and AUPRC are pooled out-of-fold metrics from the
separate [threshold analysis](results/threshold_metrics.json), rather than subject means;
AUPRC is computed as average precision. The XGBoost threshold pass omits the benchmark's sample
weights, so its AUROC/AUPRC describe a separate fit. CNN threshold metrics were not committed.*

<p align="center"><img src="outputs/figures/binary_model_comparison.png" width="560" alt="Binary LOSO accuracy for four feature models, with error bars across subjects"></p>

*Random Forest has the highest mean accuracy, but the four feature models are not significantly
different (Friedman p = 0.806; all Holm-corrected pairwise p = 1.0). Error bars show standard
deviation across subjects, not confidence intervals. The RF bootstrap 95% CI for mean accuracy
is [0.860, 0.960], from [stats.json](results/stats.json).*

The pooled RF operating point selected by Youden J has threshold 0.454, sensitivity 0.902,
specificity 0.913, PPV 0.850, and NPV 0.945. It is selected and summarized on the same pooled
out-of-fold predictions, so it is an exploratory operating point rather than an independently
validated deployment threshold.

### Three-class classification and confusion matrices

| Model | Accuracy | F1 (macro) |
| --- | ---: | ---: |
| Logistic Regression | 0.670 | 0.613 |
| LightGBM | 0.658 | 0.568 |
| Random Forest | 0.637 | 0.535 |
| XGBoost | 0.633 | 0.552 |
| 1D-CNN | 0.626 | 0.543 |

*Subject means from [metrics.json](results/metrics.json): adding amusement makes the task substantially
harder. These full-benchmark values are distinct from the matched-window leakage comparison below.*

<table>
<tr>
<td align="center"><img src="outputs/figures/binary_confusion.png" width="340" alt="Row-normalized binary Random Forest confusion matrix"><br><strong>Binary: Random Forest</strong><br>Stress recall is about 0.87 at the classifier's default decision rule.</td>
<td align="center"><img src="outputs/figures/multiclass_confusion.png" width="340" alt="Row-normalized three-class Logistic Regression confusion matrix"><br><strong>Three-class: Logistic Regression</strong><br>Baseline and amusement are frequently confused.</td>
</tr>
</table>

*Rows are true classes and columns are predictions, normalized per true class and pooled across
LOSO folds. These matrices use the default classifier decisions, not the Youden J threshold above.*

## What the analyses show

### Subject leakage

<p align="center"><img src="outputs/figures/binary_optimism_gap.png" width="520" alt="Matched-window binary accuracy: LOSO 0.907 versus subject-mixed five-fold 0.964"></p>

*On the same non-overlapping windows, subject-mixed five-fold testing raises pooled binary accuracy
from 0.907 (LOSO) to 0.964, a 5.7 percentage-point gap. Three-class pooled accuracy rises from
0.658 to 0.792, a 13.3-point gap. These are matched-window scores from
[metrics.json](results/metrics.json), not the full-benchmark subject means.*

### Feature ablation: motion and autonomic signals

<p align="center"><img src="outputs/figures/ablation.png" width="560" alt="Random Forest binary LOSO accuracy for six feature subsets"></p>

*Removing all motion features changes mean RF accuracy from 0.913 to 0.901; HRV plus EDA reaches
0.890. Motion alone reaches 0.885, so movement is predictive too. This supports a physiological
contribution without establishing freedom from confounding. Error bars are subject standard
deviations; see [ablation.csv](results/ablation.csv).*

### Chest versus wrist

<p align="center"><img src="outputs/figures/chest_vs_wrist.png" width="520" alt="Same-model Random Forest binary LOSO accuracy for chest and wrist sensors"></p>

*Using the same RF model, chest accuracy is 0.913 and wrist accuracy is 0.893. Wrist XGBoost reaches
0.906 in [wrist.json](results/wrist.json). The roughly two-point RF difference is descriptive;
15 subjects and wide uncertainty do not establish sensor equivalence.*

### Cross-dataset transfer

<p align="center"><img src="outputs/figures/cross_dataset.png" width="560" alt="Balanced accuracy within WESAD and Non-EEG and in both transfer directions"></p>

*On a separate 18-feature space shared by the devices, within-dataset balanced accuracy is 0.864
for WESAD and 0.699 for Non-EEG. Transfer reaches 0.573 from WESAD to Non-EEG and 0.500 in reverse;
see [cross_dataset.json](results/cross_dataset.json). These are balanced accuracies, not headline
accuracies from the 58-feature chest benchmark. One confounded pair cannot separate device and
domain shift from differences in stressor and label definitions; stronger generalization claims
need at least three corpora with matched stress constructs.*

### SHAP explainability

<p align="center"><img src="outputs/figures/shap_beeswarm.png" width="560" alt="Global SHAP feature importance and signed contributions for the gradient-boosted model"></p>

*The full-data gradient-boosted model emphasizes motion, heart-rate level, skin-conductance responses,
and respiration. The beeswarm shows signed feature contributions and feature values; the dashboard
uses [global mean absolute SHAP importance](results/shap_top_features.csv). This explains a model
fit on all data, rather than providing held-out evidence of causality or individual reliability.*

### Probability calibration

<!-- AUTOGEN:calibration START -->
| Evaluation | ECE | MCE | Brier |
| --- | :-: | :-: | :-: |
| Subject-mixed 5-fold, non-overlapping | 0.085 | 0.290 | 0.077 |
| LOSO, matched non-overlapping | 0.090 | 0.256 | 0.144 |
| LOSO, all windows | 0.070 | 0.160 | 0.136 |
| LOSO, all windows + isotonic | 0.025 | 0.271 | 0.129 |
<!-- AUTOGEN:calibration END -->

*Binary RF calibration from [calibration.json](results/calibration.json). Lower ECE, MCE, and Brier
are better. ECE/MCE use 15 confidence bins; Brier scores the probability of stress. The subject-mixed
and matched LOSO rows use identical non-overlapping windows. The all-window LOSO and isotonic rows
use the full pooled out-of-fold set. Isotonic reduces ECE, while maximum bin error does not improve.*

<p align="center"><img src="outputs/figures/calibration_reliability.png" width="520" alt="Reliability diagram for subject-mixed, full LOSO, and isotonic-recalibrated predictions"></p>

*The existing plot shows confidence against accuracy: full LOSO ECE falls from 0.070 to 0.025 after
isotonic recalibration fit only on training subjects' out-of-fold probabilities. Its subject-mixed
curve uses fewer, non-overlapping windows; assess leakage on the matched table rows. The matched
per-subject Brier gap is +0.066 (95% bootstrap CI [0.035, 0.106], paired Wilcoxon p < 0.001).*

### Personalization through probability recalibration

<!-- AUTOGEN:personalization START -->
| Recalibration / requested enrollment budget | ECE | Brier |
| --- | :-: | :-: |
| None (LOSO) | 0.146 | 0.073 |
| Global (training subjects) | 0.108 | 0.074 |
| Per-subject, budget 5 | 0.097 | 0.061 |
| Per-subject, budget 10 | 0.071 | 0.059 |
| Per-subject, budget 20 | 0.069 | 0.058 |
<!-- AUTOGEN:personalization END -->

*Mean per-subject scores on a reserved evaluation half from
[personalization.json](results/personalization.json). Enrollment uses the other half of non-overlapping
windows; the base classifier is not retrained. These averages use a different evaluation set and
aggregation from the pooled calibration table above.*

<p align="center"><img src="outputs/figures/personalization.png" width="520" alt="Mean per-subject ECE versus requested enrollment budget, compared with no and global recalibration"></p>

*The committed curve improves from ECE 0.146 without recalibration to 0.108 globally and 0.069 at
the requested 20-window budget. Budget labels are requests, not verified counts: the current
class-balanced sampler draws 4 windows for a request of 5 when both classes are available and can
draw fewer if a class has too few windows. The stored results are retained without regeneration.*

## Run and reproduce

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
make demo
```

`make demo` is an offline synthetic smoke check; its near-separable signals provide no scientific
evidence. The dashboard displays the committed WESAD snapshot independently of that check.

For real-data reproduction, obtain the datasets with `make wesad` and `make data`, following
[README.md](README.md), then run `make reproduce`. On macOS, XGBoost and LightGBM
also need OpenMP (`brew install libomp`). This regenerates scientific outputs and updates the
README's calibration and personalization tables through `scripts/update_readme_tables.py`.

[results/README.md](results/README.md) describes the fixed snapshot and provenance.
`requirements.lock` records its published environment; newer NeuroKit2 versions can change wrist
and transfer results, as recorded in commit `61d0d2c`. The dashboard setup is in
[frontend/README.md](frontend/README.md), module structure in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
and contribution checks in [CONTRIBUTING.md](CONTRIBUTING.md).

## Tech stack

| Area | Tools |
| --- | --- |
| Modelling | scikit-learn, XGBoost, LightGBM, PyTorch |
| Signal processing | NeuroKit2, SciPy |
| Explainability | SHAP |
| Dashboard | React, TypeScript |
| Tooling | GitHub Actions, ruff, mypy, pytest |

*The Python pipeline produces research artifacts; the React dashboard presents committed JSON.*

## Limitations

- 15 subjects, lab-induced stress. Underpowered, wide CIs. No clinical claim.
- Ablation, calibration, and personalization are exploratory, not multiplicity-corrected.
- The 1D-CNN is a small baseline, not a fair test of deep learning.
- Cross-dataset uses one confounded pair. Illustrative, not conclusive.
- Calibration improvements do not validate a clinical alert threshold or deployment in other populations.

## Future work

- A third corpus (SWELL / AffectiveROAD) for leave-one-dataset-out generalization.
- Real-world, non-lab stress data beyond the 15-subject benchmark.
- Real-time streaming inference from a live wearable.

## Ethics & data use

- Physiological signals are sensitive personal data.
- This is a research benchmark, not a product.
- Data minimization: collect and keep only what an analysis needs.
- No surveillance: do not monitor or penalize people without informed consent.
- Datasets keep their own licenses and are not redistributed here.

## Scientific references

Dataset and method attribution retained with the project:

- Schmidt, Reiss, Duerichen, Marberger, and Van Laerhoven. "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection." ICMI, 2018.
- Birjandtalab, Cogan, Pouyan, and Nourani. "A Non-EEG Dataset for Assessment of Neurological Status." IEEE BHI / PhysioNet, 2016.
- Task Force of the ESC and NASPE. "Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use." Circulation, 1996.
- Guo, Pleiss, Sun, and Weinberger. "On Calibration of Modern Neural Networks." ICML, 2017.
- Vickers and Elkin. "Decision Curve Analysis: A Novel Method for Evaluating Prediction Models." Medical Decision Making, 2006. The committed [decision-curve figure](outputs/figures/calibration_decision_curve.png) remains available.
- Lundberg and Lee. "A Unified Approach to Interpreting Model Predictions." NeurIPS, 2017.
- Bhanushali et al. "Stress Classification and Personalization: Getting the Most out of the Least." arXiv:2107.05666, 2021.
- Vos, Trinh, Sarnyai, and Rahimi Azghadi. "Generalizable Machine Learning for Stress Monitoring from Wearable Devices: A Systematic Literature Review." International Journal of Medical Informatics 173, 105026, 2023.
- Oliver and Dakshit. "Cross-Modality Investigation on WESAD Stress Classification." arXiv:2502.18733, 2025.
- Benchekroun et al. "Cross Dataset Analysis for Generalizability of HRV-Based Stress Detection Models." Sensors 23(4), 1807, 2023.
- Prajod, Mahesh, and André. "Stressor Type Matters! Exploring Factors Influencing Cross-Dataset Generalizability of Physiological Stress Detection." ICMI Companion, 2024.
- Vos et al. "Ensemble Machine Learning Model Trained on a New Synthesized Dataset Generalizes Well for Stress Prediction Using Wearable Devices." Journal of Biomedical Informatics, 2023.

## License and citation

[MIT License](LICENSE). Cite the software using [CITATION.cff](CITATION.cff).
