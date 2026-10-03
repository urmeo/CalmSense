# CalmSense Dashboard

React and TypeScript dashboard for the committed CalmSense research results. Static hosting;
no backend. [Live demo](https://urmeo.github.io/CalmSense/).

## Files

```text
src/
├── App.tsx              Routes and theme state
├── pages/               Six dashboard pages
├── components/          Shared charts, cards, and error boundary
│   └── layout/          Sidebar and header
├── data/                Exported results and signal snapshots
├── types/               Research result types
├── index.tsx            React entry point
└── index.css            Theme and shared styles
public/                  Icons and web manifest
```

## Pages

| Page | Shows |
| --- | --- |
| Dashboard | LOSO results and within-subject optimism gap |
| Signal Explorer | Real WESAD chest signals across three conditions |
| Explainability | Global SHAP importance and motion confound |
| Model Comparison | Model accuracy, wrist-only, and transfer results |
| Calibration | Reliability and decision curves; available when calibration results exist |
| About | Project overview, methods, and references |

## Development

From `frontend/`:

```bash
npm ci
npm run dev      # http://localhost:5173
npm run build    # TypeScript check and production build
npm run preview  # Preview the production build
```

CI builds the dashboard. The Pages workflow publishes `dist/` under `/CalmSense/` with a
fallback for direct page links.

## Dashboard data

The dashboard reads `src/data/results.json` and `src/data/signals.json`. To update these
snapshots after rerunning experiments, run from the repository root:

```bash
python scripts/build_dashboard_data.py  # Export committed experiment results
python scripts/export_signals.py        # Export real WESAD signal slices; requires raw data
```
