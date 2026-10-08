import typer

from anomaly_detector.baseline.xgb_tabular import train_eval_baseline


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
        graph_features=True,
        model_name="xgb_tabular_graph",
    )


if __name__ == "__main__":
    typer.run(main)
