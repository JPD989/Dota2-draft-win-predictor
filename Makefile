PYTHON   = .venv/bin/python3
DATA_DIR = data/raw
DRIFT_DIR = data/patch-742
TRAIN_PATCH ?= 7.41
TEST_PATCH  ?= 7.42
RESULTS  = results
TAG     ?=
TAG_ARG  = $(if $(TAG),--tag $(TAG),)

.PHONY: venv install install-all collect collect-200k collect-1m collect-drift data validate heroes \
        artifacts eda test test-all train-baselines train-transformer train-gnn \
        benchmark calibrate drift drift-patch app figures clean-data clean

# ---------------------------------------------------------------- setup
venv:
	python3 -m venv .venv

install: venv
	.venv/bin/pip install -e ".[dev]"

# everything: dev + Streamlit + W&B + torch (CPU wheel is enough locally)
install-all: venv
	.venv/bin/pip install -e ".[dev,app,tracking,torch]"

# ---------------------------------------------------------------- data
collect:
	$(PYTHON) src/data/collect_matches.py --target $(TARGET) --out $(DATA_DIR)

collect-200k:
	$(PYTHON) src/data/collect_matches.py --target 200000 --out $(DATA_DIR)

collect-1m:
	$(PYTHON) src/data/collect_matches.py --target 1000000 --out $(DATA_DIR) --source explorer --patch 7.41 --id-stride 8

# second patch for the drift study; kept outside DATA_DIR so it never enters the benchmark split
collect-drift:
	$(PYTHON) src/data/collect_matches.py --target 110000 --out $(DRIFT_DIR) --source explorer --patch $(TEST_PATCH) --id-stride 10

data: collect-1m validate

validate:
	$(PYTHON) src/data/validate.py --data $(DATA_DIR)

heroes:
	$(PYTHON) scripts/fetch_hero_constants.py --out data/artifacts

# train-split stats for the app (vocab, synergy/counter, win rates, meta)
artifacts:
	$(PYTHON) scripts/build_artifacts.py --data $(DATA_DIR) --out data/artifacts

# ---------------------------------------------------------------- exploratory analysis
# executes headless into results/eda/
eda:
	mkdir -p $(RESULTS)/eda
	.venv/bin/jupyter nbconvert --to notebook --execute notebooks/eda.ipynb \
		--output-dir $(RESULTS)/eda --output eda_executed.ipynb

# ---------------------------------------------------------------- tests
test:
	$(PYTHON) -m pytest -m "not slow" -q

test-all:
	$(PYTHON) -m pytest -q

# ---------------------------------------------------------------- models + evaluation
# Ablations: make train-transformer TAG=rec ARGS="--pick-dropout 0.15"
train-baselines:
	$(PYTHON) scripts/train_baselines.py --data $(DATA_DIR) $(TAG_ARG) $(ARGS)

train-transformer:
	$(PYTHON) scripts/train_transformer.py --data $(DATA_DIR) $(TAG_ARG) $(ARGS)

train-gnn:
	$(PYTHON) scripts/train_gnn.py --data $(DATA_DIR) $(TAG_ARG) $(ARGS)

benchmark:
	$(PYTHON) scripts/run_benchmark.py --data $(DATA_DIR) $(TAG_ARG) $(ARGS)

calibrate:
	$(PYTHON) scripts/calibration_analysis.py --data $(DATA_DIR) $(TAG_ARG) $(ARGS)

drift:
	$(PYTHON) scripts/patch_drift.py --data $(DATA_DIR) --mode time --models logreg xgboost transformer gnn $(ARGS)

# train on TRAIN_PATCH (DATA_DIR), test on TEST_PATCH (DRIFT_DIR)
drift-patch:
	$(PYTHON) scripts/patch_drift.py --data $(DATA_DIR) --extra-data $(DRIFT_DIR) --mode patch \
		--train-patch $(TRAIN_PATCH) --test-patch $(TEST_PATCH) --models logreg xgboost transformer gnn $(ARGS)

# ---------------------------------------------------------------- demo
app:
	.venv/bin/streamlit run app/streamlit_app.py

# ---------------------------------------------------------------- figures
# regenerate every figure of README.md and docs/report.md from results/ into docs/figures/
figures:
	$(PYTHON) scripts/make_report_figures.py --results $(RESULTS) --out docs/figures --data $(DATA_DIR) --drift-data $(DRIFT_DIR)

# ---------------------------------------------------------------- cleanup
# Deletes the collected matches (rate-limited API budget): requires CONFIRM=1.
clean-data:
	@if [ "$(CONFIRM)" != "1" ]; then \
		echo "Refusing: this deletes all shards in $(DATA_DIR). Re-run with CONFIRM=1."; exit 1; fi
	rm -f $(DATA_DIR)/matches_*.parquet $(DATA_DIR)/checkpoint.json

clean:
	@echo "'make clean' was renamed to 'make clean-data' (requires CONFIRM=1)."
