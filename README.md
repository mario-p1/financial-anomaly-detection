# Graph Neural Networks for Financial Anomaly Detection on FiGraph
A Graph Neural Network classifier for detecting financial anomalies at public companies, trained on the [FiGraph dataset](https://github.com/XiaoguangWang23/FiGraph) and compared against XGBoost and MLP baselines.

**Methodology, experiments and results are written up in [REPORT.md](REPORT.md).**

## Project Setup
### Tooling & Reproducibility
- uv for dependency management
- make for shortcuts to common tasks
- ruff for linting and formatting (run with `make format`)
- MLflow tracks every experiment, dataset, and metric.
- Dev container (`.devcontainer/`)

### Quickstart
Use the following commands to set up the project environment and download the necessary datasets:

```bash
# Synchronize the project environment
uv sync

# Download required datasets
make download_datasets
```

## Reproducing the Results
All experiments are tracked with MLflow and are fully reproducible.
The measured metrics for each model are in [REPORT.md](REPORT.md#model-performance).
The most common commands are listed below:

```bash
# Train and evaluate every model on both test years (2021 and 2022)
make all_models

# Tabular XGBoost baseline
make tabular_2021
make tabular_2022

# Graph-enhanced XGBoost baseline
make tabgraph_2021
make tabgraph_2022

# MLP classifier
make mlp_2021
make mlp_2022

# GraphSAGE (SAGEConv) classifier
make sageconv_2021
make sageconv_2022

# GATv2 classifier
make gatv2_2021
make gatv2_2022

# Tune tabular baseline
make tabular_hypopt

# Tune graph-enhanced baseline
make tabgraph_hypopt

# To run Optuna dashboard
uv run optuna-dashboard sqlite:///optuna_xgboost.sqlite3

# Visualize TensorBoard logs
uv run tensorboard --logdir tensorboard_logs

# View all run MLflow experiments
uv run mlflow server
```

## Project Structure
```
data_files/
    figraph_raw/            # FiGraph CSVs (downloaded via `make download_datasets`)

01_eda.ipynb
02_graph_eda.ipynb
figraph_tabular_columns.csv

anomaly_detector/
    data_commons.py
    eda_utils.py
    metrics.py
    utils.py
    baseline/               # XGBoost baselines and Optuna tuning
        dataset.py
        xgb_tabular.py
        xgb_tabular_graph.py
        xgboost_hyperoptim.py
    graph/                  # Graph construction and graph-derived features
        metapath_graph.py
        node_features.py
        nx_graph.py
        pyg_graph.py
    gnn_classifier/         # GNN and MLP classifiers
        dataset.py
        gatv2_classifier.py
        gnn_classifier_node_emb_base.py
        mlp_classifier.py
        model_utils.py
        sageconv_classifier.py
        train.py
```
