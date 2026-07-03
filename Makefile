.PHONY: help install-dev test lint format reproduce demo wesad data

help:
	@echo "CalmSense - subject-independent stress detection"
	@echo ""
	@echo "  install-dev    Editable install with dev tools"
	@echo "  demo           Run the pipeline on synthetic data (no download)"
	@echo "  wesad          Download WESAD (~2 GB, primary dataset; needed for reproduce)"
	@echo "  data           Download PhysioNet Non-EEG (needed for cross-dataset in reproduce)"
	@echo "  reproduce      Regenerate every result and figure (needs wesad + data)"
	@echo "  test           Run tests"
	@echo "  lint           Lint with ruff"
	@echo "  format         Format with ruff"
	@echo ""
	@echo "  Frontend dashboard: see frontend/README.md (npm run dev / build)"

install-dev:
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
