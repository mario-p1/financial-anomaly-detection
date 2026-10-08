FIGRAPH_BASE_URL := https://raw.githubusercontent.com/XiaoguangWang23/FiGraph/refs/heads/main/data
FIGRAPH_YEARS := 2014 2015 2016 2017 2018 2019 2020 2021 2022

TABULAR_CMD := uv run python -m anomaly_detector.baseline.xgb_tabular \
	--learning-rate 0.3 \
	--max-depth 9 \
	--min-child-weight 1 \
	--subsample 0.9 \
	--colsample-bytree 0.9 \
	--n-estimators 1000 \
	--early-stopping-rounds 50

TABGRAPH_CMD := uv run python -m anomaly_detector.baseline.xgb_tabular_graph \
	--learning-rate 0.3 \
	--max-depth 9 \
	--min-child-weight 1 \
	--subsample 0.9 \
	--colsample-bytree 0.9 \
	--n-estimators 1000 \
	--early-stopping-rounds 50

SAGECONV_CMD := uv run python -m anomaly_detector.gnn_classifier.train \
	--layer-type sageconv \
	--epochs 500 \
	--learning-rate 0.001 \
	--hidden-channels 128 \
	--dropout 0.3

GATV2_CMD := uv run python -m anomaly_detector.gnn_classifier.train \
	--layer-type gatv2 \
	--epochs 500 \
	--learning-rate 0.001 \
	--hidden-channels 128 \
	--dropout 0.3 \
	--heads 2

MLP_CMD := uv run python -m anomaly_detector.gnn_classifier.train \
	--layer-type mlp \
	--epochs 500 \
	--learning-rate 0.001 \
	--hidden-channels 128 \
	--dropout 0.3

YEARS_2021 := --train-year-start 2014 \
	--train-year-end 2019 \
	--valid-year-start 2020 \
	--valid-year-end 2020 \
	--test-year-start 2021 \
	--test-year-end 2021

YEARS_2022 := --train-year-start 2014 \
	--train-year-end 2020 \
	--valid-year-start 2021 \
	--valid-year-end 2021 \
	--test-year-start 2022 \
	--test-year-end 2022

.PHONY: download_datasets
download_datasets:
	for year in $(FIGRAPH_YEARS); do \
		wget $(FIGRAPH_BASE_URL)/$$year/edges$$year.csv \
			-O data_files/figraph_raw/edges$$year.csv; \
		wget $(FIGRAPH_BASE_URL)/$$year/ListedCompanyFeatures772_$$year.csv \
			-O data_files/figraph_raw/ListedCompanyFeatures772_$$year.csv; \
	done

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check --extend-select I --fix .

.PHONY: tabular_2021
tabular_2021:
	# Tabular baseline
	$(TABULAR_CMD) $(YEARS_2021)

.PHONY: tabular_2022
tabular_2022:
	# Tabular baseline
	$(TABULAR_CMD) $(YEARS_2022)

.PHONY: tabular_hypopt
tabular_hypopt:
	# Tabular baseline hyperparameter optimization
	uv run python -m anomaly_detector.baseline.xgboost_hyperoptim --optim-model tabular

.PHONY: tabgraph_2021
tabgraph_2021:
	# Tabular with graph features baseline
	$(TABGRAPH_CMD) $(YEARS_2021)

.PHONY: tabgraph_2022
tabgraph_2022:
	# Tabular with graph features baseline
	$(TABGRAPH_CMD) $(YEARS_2022)

.PHONY: tabgraph_hypopt
tabgraph_hypopt:
	# Tabular with graph features hyperparameter optimization
	uv run python -m anomaly_detector.baseline.xgboost_hyperoptim --optim-model graph_enhanced

.PHONY: sageconv_2021
sageconv_2021:
	$(SAGECONV_CMD) $(YEARS_2021)

.PHONY: sageconv_2022
sageconv_2022:
	$(SAGECONV_CMD) $(YEARS_2022)

.PHONY: gatv2_2021
gatv2_2021:
	$(GATV2_CMD) $(YEARS_2021)

.PHONY: gatv2_2022
gatv2_2022:
	$(GATV2_CMD) $(YEARS_2022)

.PHONY: mlp_2021
mlp_2021:
	$(MLP_CMD) $(YEARS_2021)

.PHONY: mlp_2022
mlp_2022:
	$(MLP_CMD) $(YEARS_2022)

.PHONY: all_models
all_models: tabular_2021 tabular_2022 \
	tabgraph_2021 tabgraph_2022 \
	mlp_2021 mlp_2022 \
	sageconv_2021 sageconv_2022 \
	gatv2_2021 gatv2_2022
