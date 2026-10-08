from functools import partial
from typing import Literal

import optuna
import typer
from xgboost import XGBClassifier

from anomaly_detector.baseline.dataset import (
    TabularDataset,
    build_baseline_dataset,
)
from anomaly_detector.baseline.xgb_tabular import DROP_COLUMNS, ONE_HOT_COLUMNS
from anomaly_detector.metrics import compute_metrics


def objective(
    trial: optuna.Trial,
    dataset: TabularDataset,
    optim_metric: str,
) -> float:
    min_valid_year = min(dataset.get_valid_years)

    n_pos = dataset.valid_y_dict[min_valid_year].sum()
    n_neg = len(dataset.valid_y_dict[min_valid_year]) - n_pos

    scale_pos_weight = n_neg / n_pos

    fixed_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
        "n_estimators": 2000,
        "scale_pos_weight": scale_pos_weight,
    }
    tuning_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    for k, v in fixed_params.items():
        trial.set_user_attr(k, v)

    model = XGBClassifier(
        **tuning_params,
        **fixed_params,
        random_state=42,
    )

    model.fit(
        dataset.train_x,
        dataset.train_y,
        eval_set=[
            (
                dataset.valid_x_dict[min_valid_year],
                dataset.valid_y_dict[min_valid_year],
            ),
        ],
        verbose=False,
    )

    trial.set_user_attr("best_iteration", model.best_iteration)

    valid_preds = model.predict_proba(
        dataset.valid_x_dict[min_valid_year])[:, 1]

    valid_metrics = compute_metrics(
        dataset.valid_y_dict[min_valid_year], valid_preds)

    return valid_metrics[optim_metric]


def main(
    train_year_start: int = 2014,
    train_year_end: int = 2019,
    valid_year_start: int = 2020,
    valid_year_end: int = 2021,
    n_trials: int = 100,
    optim_metric: str = "au_prc",
    optim_model: Literal["tabular", "graph_enhanced"] = "graph_enhanced",
    continue_study: str | None = None,
):
    print("XGBOOST Hyperparameter Optimization")

    print("Building dataset...")
    dataset = build_baseline_dataset(
        train_year_start=train_year_start,
        train_year_end=train_year_end,
        valid_year_start=valid_year_start,
        valid_year_end=valid_year_end,
        drop_columns=DROP_COLUMNS,
        one_hot_columns=ONE_HOT_COLUMNS,
        graph_features=(optim_model == "graph_enhanced"),
    )

    print(f"Starting hyperparameter optimization for {optim_model} model.")

    s = optuna.create_study(
        storage="sqlite:///optuna_xgboost.sqlite3",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=continue_study,
        load_if_exists=(continue_study is not None),
    )
    s.set_user_attr("optim_model", optim_model)
    s.set_user_attr("optim_metric", optim_metric)
    s.set_user_attr("train_period", f"{train_year_start}-{train_year_end}")
    s.set_user_attr("valid_period", f"{valid_year_start}-{valid_year_end}")

    s.optimize(
        partial(objective, dataset=dataset, optim_metric=optim_metric),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("Hypereparameter tuning results:")
    print(f"Best parameters: {s.best_params}")
    print(f"Best validation {optim_metric}: {s.best_value}")


if __name__ == "__main__":
    typer.run(main)
