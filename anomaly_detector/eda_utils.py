from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import homophily
from tqdm import tqdm

from anomaly_detector.graph.metapath_graph import build_metapath_graph
from anomaly_detector.graph.pyg_graph import build_yearly_hetero_graph


def describe_categorical(
    df: pd.DataFrame,
    columns: list[str],
    decimal_places: int = 2,
) -> pd.DataFrame:
    # Filter columns to only include those present in the DataFrame
    cols = [col for col in df.columns if col in columns]

    if set(cols) != set(columns):
        missing_columns = set(columns) - set(cols)
        print(
            f"Warning: The following columns are not in the "
            f"DataFrame and will be ignored: {missing_columns}",
        )

    desc_df = pd.DataFrame()
    desc_df["column"] = cols
    desc_df["count"] = [df[col].count() for col in cols]
    desc_df["missing"] = df.shape[0] - desc_df["count"]
    desc_df["missing_percentage"] = desc_df["missing"] * \
        1.0 / df.shape[0] * 100

    desc_df["unique_values"] = [df[col].nunique() for col in cols]

    value_counts_per_column = {col: df[col].value_counts() for col in cols}

    desc_df["mode"] = [value_counts_per_column[col].index[0] for col in cols]
    desc_df["mode_count"] = [value_counts_per_column[col].iloc[0]
                             for col in cols]
    desc_df["mode_percentage"] = desc_df["mode_count"] / desc_df["count"] * 100

    desc_df["mode_2"] = [
        value_counts_per_column[col].index[1]
        if len(value_counts_per_column[col]) > 1
        else None
        for col in cols
    ]
    desc_df["mode_2_count"] = [
        value_counts_per_column[col].iloc[1]
        if len(value_counts_per_column[col]) > 1
        else None
        for col in cols
    ]
    desc_df["mode_2_percentage"] = desc_df["mode_2_count"] / \
        desc_df["count"] * 100

    desc_df = desc_df.round(decimal_places)

    return desc_df


def describe_numerical(
    df: pd.DataFrame,
    columns: list[str],
    decimal_places: int = 2,
    percentiles: list[float] | None = None,
    include_missingness_metrics: bool = True,
    include_outlier_metrics: bool = True,
    include_skewness_metrics: bool = True,
) -> pd.DataFrame:

    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]

    # Filter columns to only include those present in the DataFrame
    cols = [col for col in df.columns if col in columns]

    if set(cols) != set(columns):
        missing_columns = set(columns) - set(cols)
        print(
            f"Warning: The following columns are not in the DataFrame"
            f" and will be ignored: {missing_columns}",
        )

    desc_df = pd.DataFrame()
    desc_df["column"] = cols
    desc_df["count"] = [df[col].count() for col in cols]

    if include_missingness_metrics:
        desc_df["missing"] = df.shape[0] - desc_df["count"]
        desc_df["missing_percentage"] = desc_df["missing"] * \
            1.0 / df.shape[0] * 100

    desc_df["unique_values"] = [df[col].nunique() for col in cols]

    desc_df["mean"] = [df[col].mean() for col in cols]
    desc_df["std"] = [df[col].std() for col in cols]

    desc_df["min"] = [df[col].min() for col in cols]

    for q in percentiles:
        desc_df[f"percentile_{int(q * 100)}"] = [df[col].quantile(q)
                                                 for col in cols]

    desc_df["max"] = [df[col].max() for col in cols]

    quantiles = {
        col: {
            "q1": df[col].quantile(0.25),
            "q2": df[col].median(),
            "q3": df[col].quantile(0.75),
        }
        for col in cols
    }

    if include_skewness_metrics:
        desc_df["pearson_skewness"] = [df[col].skew() for col in cols]

        desc_df["bowley_skewness"] = [
            (quantiles[col]["q3"] + quantiles[col]
             ["q1"] - 2 * quantiles[col]["q2"])
            / (quantiles[col]["q3"] - quantiles[col]["q1"])
            if quantiles[col]["q3"] - quantiles[col]["q1"] != 0
            else None
            for col in cols
        ]

    if include_outlier_metrics:
        desc_df["iqr_outliers_percentage"] = [
            df[col][
                (
                    df[col]
                    > quantiles[col]["q3"]
                    + 1.5 * (quantiles[col]["q3"] - quantiles[col]["q1"])
                )
                | (
                    df[col]
                    < quantiles[col]["q1"]
                    - 1.5 * (quantiles[col]["q3"] - quantiles[col]["q1"])
                )
            ].count()
            / df[col].count()
            * 100
            for col in cols
        ]

    desc_df = desc_df.round(decimal_places)

    return desc_df


def draw_count_bar_plot(
    counts: pd.Series,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scale: Literal["linear", "log"] = "linear",
    legend_dict: dict[str | int, str] | None = None,
):
    _, ax = plt.subplots(figsize=(8, 6))
    counts.plot.bar(
        ax=ax,
        width=0.7,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=plt.cm.tab10.colors,
    )
    ax.set_yscale(scale)

    plt.xticks(rotation=0)

    if legend_dict is not None:
        ax.legend(
            handles=ax.containers[0].patches,
            labels=[legend_dict.get(index, index) for index in counts.index],
        )

    ax.bar_label(
        ax.containers[0],
        labels=[f"{p}: {p / counts.sum():.0%}" for p in counts],
        label_type="center",
        color="white",
        fontsize=8,
    )
    plt.show()


def create_ccdf(values: pd.Series) -> pd.Series:
    # Sort in descending order
    counts = values.value_counts().sort_index(ascending=False)

    # Calculate CCDF
    ccdf = counts.cumsum() / counts.sum()

    # Set index and name for the series
    ccdf.index.name = "k"
    ccdf = ccdf.rename("ccdf")

    # Reverse the order have the CCDF in ascending order of k
    return ccdf.iloc[::-1]


def create_cdf(values: pd.Series) -> pd.Series:
    # Sort in ascending order
    counts = values.value_counts().sort_index(ascending=True)

    # Calculate CDF
    cdf = counts.cumsum() / counts.sum()

    # Set index and name for the series
    cdf.index.name = "k"
    cdf = cdf.rename("cdf")

    return cdf


def ccdf_plot(
    df: pd.DataFrame,
    column: str,
    group_by: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    for group_by_value, group_df in df.groupby(group_by):
        per_year = [create_ccdf(g[column])
                    for _, g in group_df.groupby("year")]

        # Align the years on their shared degree grid
        band = pd.concat(per_year, axis=1).sort_index().bfill().fillna(0)

        low, high = band.min(axis=1), band.max(axis=1)

        ax.fill_between(band.index, low, high, alpha=0.25, linewidth=0)
        ax.plot(band.index, band.median(axis=1),
                linewidth=2, label=group_by_value)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("$P(K > k)$")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.4)
    ax.legend(title=group_by)

    return ax


def cdf_plot(
    df: pd.DataFrame,
    column: str,
    group_by: str,
    ax: plt.Axes | None = None,
    xscale: Literal["linear", "log"] = "linear",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    for group_by_value, group_df in df.groupby(group_by):
        per_year = [create_cdf(g[column]) for _, g in group_df.groupby("year")]

        # Align the years on their shared degree grid
        band = pd.concat(per_year, axis=1).sort_index().ffill().fillna(0)

        low, high = band.min(axis=1), band.max(axis=1)

        ax.fill_between(band.index, low, high, alpha=0.25, linewidth=0)
        ax.plot(band.index, band.median(axis=1),
                linewidth=2, label=group_by_value)

    ax.set_xscale(xscale)
    ax.set_ylabel("$P(K \\leq k)$")
    ax.legend(title=group_by)
    return ax


def draw_stacked_count_bar_plot(
    df: pd.DataFrame,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label: Literal["percentage", "count"] = "percentage",
):
    _, ax = plt.subplots(figsize=(8, 6))
    df.plot.bar(
        ax=ax,
        stacked=True,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    plt.xticks(rotation=0)

    if label == "percentage":
        label_values = df.div(df.sum(axis=1), axis=0)
        formatter = "{:.1%}".format
    else:
        label_values = df
        formatter = "{}".format

    for container, (_, segment) in zip(ax.containers, label_values.items()):
        ax.bar_label(
            container,
            labels=[formatter(v) for v in segment],
            label_type="center",
            color="white",
            fontsize=8,
        )
    plt.show()


def homophily_permutation_scores(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    nperm: int = 100,
    seed: int = 42,
    method: Literal["node", "edge", "edge_insensitive"] = "node",
) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(nperm):
        perm = torch.randperm(y.size(0), generator=g)
        scores.append(homophily(edge_index, y[perm], method=method))
    return np.array(scores)


def compute_homophily_scores(
    tabular_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    years: list[int],
    method: Literal["node", "edge", "edge_insensitive"],
    nperm: int,
) -> pd.DataFrame:
    homophily_scores = []

    with tqdm(total=-1) as pbar:
        for year in years:
            graph, _ = build_yearly_hetero_graph(
                tabular_df=tabular_df.loc[lambda x, y=year: x["Year"] == y],
                edges_df=edges_df.loc[lambda x, y=year: x["year"] == y],
            )

            graph["L"].y = graph["L"].y.to(torch.int32)

            metapath_graph = build_metapath_graph(graph=graph)

            if pbar.total == -1:
                pbar.total = len(metapath_graph.edge_types) * len(years)

            for edge_type in metapath_graph.edge_types:
                pbar.update(1)
                edge_name = (
                    f"{metapath_graph.metapath_dict[edge_type]}"
                    if edge_type[1].startswith("metapath_")
                    else f"{edge_type}"
                )
                h_score = homophily(
                    metapath_graph[edge_type].edge_index,
                    metapath_graph["L"].y,
                    method=method,
                )
                perm_scores = homophily_permutation_scores(
                    edge_index=metapath_graph[edge_type].edge_index,
                    y=metapath_graph["L"].y,
                    nperm=nperm,
                    method=method,
                )

                p = (1 + (perm_scores >= h_score).sum()) / (1 + nperm)
                perms_mean = np.mean(perm_scores).item()

                homophily_scores.append(
                    {
                        "edge_name": edge_name,
                        "year": year,
                        "num_edges": metapath_graph[edge_type].edge_index.shape[1],
                        "h_score": h_score,
                        "h_perm_mean": perms_mean,
                        "excess_h": h_score - perms_mean,
                        "excess_h_z_score": (h_score - perms_mean)
                        / np.std(perm_scores),
                        "p": p,
                        "nperm": nperm,
                    },
                )

    return pd.DataFrame(homophily_scores).sort_values(by=["edge_name", "year"])
