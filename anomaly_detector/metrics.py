from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_threshold = -1
    mcc_at_best_threshold = -1

    if mlflow.active_run():
        mlflow.log_param("threshold_method", "best_mcc")

    for mcc_threshold in np.linspace(0, 1, 101):
        y_pred_mcc = (y_score >= mcc_threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_mcc)
        if mcc > mcc_at_best_threshold:
            mcc_at_best_threshold = mcc
            best_threshold = mcc_threshold

    return best_threshold


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    at_threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (y_score >= at_threshold).astype(int)

    return {
        "prevalence": y_true.mean(),
        "au_roc": roc_auc_score(y_true, y_score),
        "au_prc": average_precision_score(y_true, y_score),
        "at_threshold": at_threshold,
        "classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def mlflow_log_metrics(metrics: dict[str, Any], prefix: str = "custom/") -> None:
    mlflow.log_metrics(
        {
            f"{prefix}{name}": value
            for name, value in metrics.items()
            if name != "classification_report"
        },
    )
    if "classification_report" in metrics:
        mlflow.log_text(
            metrics["classification_report"],
            f"{prefix}classification_report.txt",
        )


def mlflow_evaluate(
    model_uri: str,
    x: pd.DataFrame,
    y_true: np.ndarray,
    dataset_name: str | None = None,
    threshold: float = 0.5,
    metric_prefix: str = "custom/",
) -> dict[str, Any]:
    eval_data_mlflow = mlflow.data.from_pandas(
        x.assign(true_label=y_true),
        targets="true_label",
        name=dataset_name,
    )

    mlflow.models.evaluate(
        model_uri,
        eval_data_mlflow,
        model_type="classifier",
    )

    model = mlflow.pyfunc.load_model(model_uri).get_raw_model()
    y_score = model.predict_proba(x)[:, 1]

    metrics = compute_metrics(y_true, y_score, at_threshold=threshold)
    mlflow_log_metrics(metrics, prefix=metric_prefix)

    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    print("===============================")
    print("            METRICS")
    print("===============================\n")
    print(f"Positive rate:          {metrics['prevalence']:.4f}")
    print(f"AU ROC:                 {metrics['au_roc']:.4f}")
    print(f"AU PRC:                 {metrics['au_prc']:.4f}")
    print(f"At threshold {metrics['at_threshold']}:")
    print(metrics["classification_report"])
    print(f"MCC:                    {metrics['mcc']:.4f}")
