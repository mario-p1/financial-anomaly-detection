from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from anomaly_detector.baseline.xgb_tabular import DROP_COLUMNS, ONE_HOT_COLUMNS
from anomaly_detector.data_commons import load_figraph_data, order_edge_nodes
from anomaly_detector.graph.node_features import build_node_features_preprocessor


def build_yearly_hetero_graph(
    tabular_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> tuple[HeteroData, dict[str, dict[str, int]]]:
    # Order nodes in the edges
    edges_df = order_edge_nodes(edges_df)

    # Extract node IDs from the tabular data
    tabular_companies = set(tabular_df["nodeID"])

    # Build node set
    nodes_in_edges = set(pd.concat([edges_df["id_1"], edges_df["id_2"]]))
    nodes_set = defaultdict(set)
    for node in nodes_in_edges:
        node_type = node[0]
        if node_type != "L":
            nodes_set[node_type].add(node)

    nodes_set["L"] = tabular_companies

    # Create node index mappings
    node_mappings: dict[str, dict[str, int]] = {
        node_type: {node_id: idx for idx, node_id in enumerate(sorted(ids))}
        for node_type, ids in nodes_set.items()
    }

    l_node_mappings, l_features, l_labels = build_l_node_features(tabular_df)

    node_mappings["L"] = l_node_mappings

    # Build the heterogeneous graph
    graph = HeteroData()
    for node_type, node_ids in node_mappings.items():
        graph[node_type].num_nodes = len(node_ids)

    # Assign features and labels to listed companies
    graph["L"].x_df, graph["L"].y = l_features, l_labels

    endpoints: dict[tuple[str, str, str], tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], []),
    )
    for id_1, relation, id_2 in zip(
        edges_df["id_1"],
        edges_df["type"],
        edges_df["id_2"],
    ):
        src_type, tgt_type = id_1[0], id_2[0]
        source, target = endpoints[src_type, relation, tgt_type]
        source.append(node_mappings[src_type][id_1])
        target.append(node_mappings[tgt_type][id_2])

    # Add edges
    for (src_type, relation, tgt_type), (source, target) in endpoints.items():
        forward = torch.tensor(np.stack([source, target]), dtype=torch.long)
        reverse = forward.flip(0)

        if src_type == tgt_type:
            # A symmetric relation keeps both directions in one store.
            graph[src_type, relation, tgt_type].edge_index = torch.cat(
                [forward, reverse],
                dim=1,
            )
        else:
            graph[src_type, relation, tgt_type].edge_index = forward
            graph[tgt_type, relation, src_type].edge_index = reverse

    graph.metadata = {
        "year": edges_df["year"].iloc[0], "node_mappings": node_mappings}

    graph.validate()

    return graph, node_mappings


def build_l_node_features(
    companies_df: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame, torch.Tensor]:
    companies_df = companies_df.copy()

    node_mappings = {node_id: idx for idx,
                     node_id in enumerate(companies_df["nodeID"])}

    labels = companies_df["Label"].values
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    features = companies_df.drop(columns=["Label"])

    return node_mappings, features, labels_tensor


if __name__ == "__main__":
    tabular_dict, edges_dict = load_figraph_data(years=[2014, 2015])

    tabular_df = tabular_dict[2014]
    edges_df = edges_dict[2014]

    # Remove Audit edges
    edges_df = edges_df.loc[lambda x: x["type"] != "Audit"]

    graph, node_mappings = build_yearly_hetero_graph(tabular_df, edges_df)

    preprocessor = build_node_features_preprocessor(
        companies_df=tabular_df.drop(columns=["Label"]),
        one_hot_columns=ONE_HOT_COLUMNS,
        drop_columns=DROP_COLUMNS,
    )

    enc_df = preprocessor.transform(graph["L"].x_df)
    graph["L"].x = torch.tensor(enc_df.to_numpy(), dtype=torch.float)
