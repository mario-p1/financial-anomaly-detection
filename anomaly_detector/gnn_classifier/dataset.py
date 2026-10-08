from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from anomaly_detector.baseline.xgb_tabular import DROP_COLUMNS, ONE_HOT_COLUMNS
from anomaly_detector.data_commons import (
    find_leafs_edges_df,
    load_figraph_data,
    remove_nodes_in_dataframes,
)
from anomaly_detector.graph.node_features import build_node_features_preprocessor
from anomaly_detector.graph.pyg_graph import build_yearly_hetero_graph


@dataclass
class Graphs:
    train_graphs: dict[int, HeteroData]
    valid_graphs: dict[int, HeteroData]
    test_graphs: dict[int, HeteroData]


def build_gnn_graphs(
    train_year_start: int,
    train_year_end: int,
    valid_year_start: int,
    valid_year_end: int,
    test_year_start: int | None = None,
    test_year_end: int | None = None,
) -> Graphs:
    assert train_year_start <= train_year_end, (
        "train_year_start must be less than or equal to train_year_end"
    )
    assert valid_year_start <= valid_year_end, (
        "valid_year_start must be less than or equal to valid_year_end"
    )
    assert valid_year_start > train_year_end, (
        "valid_year_start must be greater than train_year_end"
    )

    if test_year_start is not None:
        assert test_year_end is not None, (
            "test_year_end must be provided if test_year_start is provided"
        )
        assert test_year_start <= test_year_end, (
            "test_year_start must be less than or equal to test_year_end"
        )
        assert test_year_start > valid_year_end, (
            "test_year_start must be greater than valid_year_end"
        )

    train_range = list(range(train_year_start, train_year_end + 1))
    valid_range = list(range(valid_year_start, valid_year_end + 1))
    test_range = (
        list(range(test_year_start, test_year_end + 1))
        if test_year_start is not None
        else []
    )

    tabular_dict, edges_dict = load_figraph_data(
        years=train_range + valid_range + test_range
    )

    train_graphs = {}
    valid_graphs = {}
    test_graphs = {}

    train_tabular_df = pd.concat([tabular_dict[year] for year in train_range])

    ct = build_node_features_preprocessor(
        train_tabular_df.drop(columns=["Label"]), ONE_HOT_COLUMNS, DROP_COLUMNS
    )

    for year in train_range + valid_range + test_range:
        tabular_df = tabular_dict[year]
        edges_df = edges_dict[year]

        # Remove Audit edges
        edges_df = edges_df.loc[lambda x: x["type"] != "Audit"].copy()

        # Rename relation types
        edges_df["type"] = (
            edges_df["type"].str.lower().str.replace(
                " ", "_").str.replace("-", "_")
        )

        # Build graph
        graph, _ = build_yearly_hetero_graph(tabular_df, edges_df)

        # Apply ct
        enc_df = ct.transform(graph["L"].x_df)
        graph["L"].x = torch.tensor(enc_df.to_numpy(), dtype=torch.float)

        if year in train_range:
            train_graphs[year] = graph
        elif year in valid_range:
            valid_graphs[year] = graph
        else:
            test_graphs[year] = graph

    return Graphs(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        test_graphs=test_graphs,
    )


def compute_pos_weight(graphs: list[HeteroData]) -> torch.Tensor:
    labels = torch.cat([graph["L"].y for graph in graphs])
    return (labels.numel() - labels.sum()) / labels.sum()
