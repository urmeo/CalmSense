.PHONY: help install-dev test lint format reproduce demo wesad data

help:
	@echo "CalmSense - subject-independent stress detection"
	@echo ""
	@echo "  install-dev    Reproducible install: pinned deps (requirements.lock) + dev tools"
	@echo "  demo           Run the pipeline on synthetic data (no download)"
	@echo "  wesad          Download WESAD (~2 GB, primary dataset; needed for reproduce)"
	@echo "  data           Download PhysioNet Non-EEG (needed for cross-dataset in reproduce)"
	@echo "  reproduce      Regenerate every result and figure (needs wesad + data)"
	@echo "  test           Run tests"
	@echo "  lint           Lint with ruff"
	@echo "  format         Format with ruff"
	@echo ""
	@echo "  Frontend dashboard: see frontend/README.md (npm run dev / build)"

# Install the exact pinned versions the results were produced with (requirements.lock),
# then the editable package and dev tools. The pins matter: the shipped model pickle is
# coupled to scikit-learn 1.6.1, so a looser install can load it with a version-mismatch warning.
install-dev:
	pip install -r requirements.lock
	pip install -e ".[dev]"

demo:
	python scripts/calibration.py --synthetic

wesad:
	python scripts/download_data.py --wesad

data:
	python scripts/download_data.py

# Regenerate every number, figure, model, and the dashboard data in order.
# Prerequisites: WESAD (make wesad) and PhysioNet Non-EEG (make data) must be downloaded first;
# on macOS, xgboost/lightgbm also need OpenMP (brew install libomp).
reproduce:
	@test -f data/raw/WESAD/S2/S2.pkl || test -f data/processed/features.parquet || \
		{ echo "ERROR: WESAD not found. Run 'make wesad' (and 'make data' for cross_dataset) first; see data/raw/README.md."; exit 1; }
	python scripts/run_experiment.py
	python scripts/ablation.py
	python scripts/wrist.py
	python scripts/cross_dataset.py
	python scripts/calibration.py
	python scripts/personalize.py
	python scripts/fill_paper_tables.py
	python scripts/tuning.py
	python scripts/stats.py
	python scripts/export_onnx.py
	python scripts/build_dashboard_data.py

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/
