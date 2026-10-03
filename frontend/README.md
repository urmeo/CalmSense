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
public/                  Icons
config/                  Package, lock, compiler, and manifest source modules
tooling.mjs              Install, audit, build, and development commands
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
node tooling.mjs install
node tooling.mjs dev      # Development server
node tooling.mjs build    # TypeScript check and production build
node tooling.mjs preview  # Preview the production build
```

Node **24** and npm are required. Configuration lives in `config/*.mjs`; commands generate
the local JSON files required by npm and TypeScript. These generated files are ignored by Git.
Edit the source modules rather than the generated files.
After editing dependencies, run `node tooling.mjs update` and review the source lock changes.

CI installs the exact locked dependencies, audits moderate or higher vulnerabilities, and builds
the dashboard. GitHub's automatic dependency discovery does not read the custom manifest.
The Pages workflow publishes `dist/` under `/CalmSense/` with a fallback for direct page links.

## Dashboard data

The dashboard imports `src/data/results.ts` and `src/data/signals.ts`. To update these
snapshots after rerunning experiments, run from the repository root:

```bash
python scripts/build_dashboard_data.py  # Export committed experiment results
python scripts/export_signals.py        # Export real WESAD signal slices; requires raw data
```
