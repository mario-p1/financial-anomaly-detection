from itertools import chain, repeat

import mlflow
import typer
from mlflow.models import infer_signature
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from anomaly_detector.baseline.dataset import build_baseline_dataset
from anomaly_detector.metrics import find_best_threshold, mlflow_evaluate, print_metrics

ONE_HOT_COLUMNS = ["IndustryCodeC"]

DROP_COLUMNS = [
    "A001109000",
    "A001206000",
    "A001207000",
    "A003107000",
    "B001304000",
    "B002200000",
    "A001127000",
    "A001128000",
    "A001226000",
    "A001227000",
    "A001228000",
    "A001229000",
    "A001230000",
    "A002128000",
    "A002127000",
    "A002211000",
    "A002210000",
    "A003112000",
    "A003112101",
    "A003112201",
    "A003112301",
    "A003111000",
    "B001216000",
    "B001211101",
    "B001211203",
    "B001305000",
    "B001302201",
    "B001306000",
    "B001307000",
    "B001308000",
    "B001400101",
    "B001500201",
    "B002000301",
    "nodeID",
    "Year",
    "RegisterLatitude",
    "RegisterLongitude",
    "B006000103",
]


def train_eval_baseline(
    learning_rate: float,
    max_depth: int,
    min_child_weight: int,
    subsample: float,
    colsample_bytree: float,
    n_estimators: int,
    early_stopping_rounds: int,
    graph_features: bool,
    model_name: str,
    train_year_start: int,
    train_year_end: int,
    valid_year_start: int,
    valid_year_end: int,
    test_year_start: int | None,
    test_year_end: int | None,
    eval_metric: str = "aucpr",
):
    dataset = build_baseline_dataset(
        train_year_start=train_year_start,
        train_year_end=train_year_end,
        valid_year_start=valid_year_start,
        valid_year_end=valid_year_end,
        test_year_start=test_year_start,
        test_year_end=test_year_end,
        drop_columns=DROP_COLUMNS,
        one_hot_columns=ONE_HOT_COLUMNS,
        graph_features=graph_features,
    )

    n_pos = dataset.valid_y_dict[valid_year_start].sum()
    n_neg = len(dataset.valid_y_dict[valid_year_start]) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=eval_metric,
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        callbacks=[
            EarlyStopping(rounds=early_stopping_rounds,
                          save_best=True, maximize=True),
        ],
    )

    print("Training model...")
    model.fit(
        dataset.train_x,
        dataset.train_y,
        eval_set=[
            (
                dataset.valid_x_dict[valid_year_start],
                dataset.valid_y_dict[valid_year_start],
            ),
        ],
        verbose=True,
    )
    print("Model training completed.")

    # Log experiment
    train_data_mlflow = mlflow.data.from_pandas(
        dataset.train_x.assign(true_label=dataset.train_y),
        targets="true_label",
        name="train",
    )
    with mlflow.start_run():
        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "learning_rate": model.learning_rate,
                "max_depth": model.max_depth,
                "min_child_weight": model.min_child_weight,
                "subsample": model.subsample,
                "colsample_bytree": model.colsample_bytree,
                "scale_pos_weight": scale_pos_weight,
                "graph_features": graph_features,
                "best_iteration": model.best_iteration,
                "train_year_start": train_year_start,
                "train_year_end": train_year_end,
                "valid_year_start": valid_year_start,
                "valid_year_end": valid_year_end,
                "model": model_name,
            },
        )

        signature = infer_signature(
            dataset.valid_x_dict,
            model.predict(dataset.valid_x_dict[valid_year_start]),
        )
        model_info = mlflow.xgboost.log_model(
            model,
            name=model_name,
            signature=signature,
        )

        mlflow.log_input(train_data_mlflow)

        best_threshold = find_best_threshold(
            dataset.valid_y_dict[valid_year_start],
            model.predict_proba(dataset.valid_x_dict[valid_year_start])[:, 1],
        )
        print(
            f"Best threshold found on validation set ({valid_year_start}): {best_threshold}",
        )
        mlflow.log_param("threshold_period", valid_year_start)
        mlflow.log_param("best_threshold", best_threshold)

        for dset, year in chain(
            zip(repeat("valid"), dataset.valid_x_dict),
            zip(repeat("test"), dataset.test_x_dict),
        ):
            if dset == "valid":
                x = dataset.valid_x_dict[year]
                y = dataset.valid_y_dict[year]
            else:
                x = dataset.test_x_dict[year]
                y = dataset.test_y_dict[year]

            with mlflow.start_run(nested=True):
                mlflow.log_param("dataset", f"{dset}_{year}")

                valid_metrics = mlflow_evaluate(
                    model_info.model_uri,
                    x,
                    y,
                    dataset_name=f"{dset}_{year}",
                    threshold=best_threshold,
                )
                print(f"Validation metrics for year: {year}")
                print_metrics(valid_metrics)

        with mlflow.start_run(nested=True):
            valid_unique_comp_metrics = mlflow_evaluate(
                model_info.model_uri,
                dataset.unique_valid_x,
                dataset.unique_valid_y,
                dataset_name=f"unique_valid_companies_{valid_year_start}-{valid_year_end}",
                threshold=best_threshold,
            )
            print("Validation metrics (unique companies):")
            print_metrics(valid_unique_comp_metrics)


def main(
    learning_rate: float = typer.Option(),
    max_depth: int = typer.Option(),
    min_child_weight: int = typer.Option(),
    subsample: float = typer.Option(),
    colsample_bytree: float = typer.Option(),
    n_estimators: int = typer.Option(),
    early_stopping_rounds: int = typer.Option(),
    train_year_start: int = typer.Option(),
    train_year_end: int = typer.Option(),
    valid_year_start: int = typer.Option(),
    valid_year_end: int = typer.Option(),
    test_year_start: int | None = typer.Option(None),
    test_year_end: int | None = typer.Option(None),
):
    train_eval_baseline(
        train_year_start=train_year_start,
        train_year_end=train_year_end,
        valid_year_start=valid_year_start,
        valid_year_end=valid_year_end,
        test_year_start=test_year_start,
        test_year_end=test_year_end,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        graph_features=False,
        model_name="xgb_tabular",
    )


if __name__ == "__main__":
    typer.run(main)
