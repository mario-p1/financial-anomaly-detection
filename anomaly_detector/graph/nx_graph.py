import networkx as nx
import pandas as pd


def build_networkx_graph(
    tabular_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    label_column: str | None = "Label",
) -> nx.Graph:
    nx_graph = nx.Graph()

    for _, row in tabular_df.iterrows():
        node_id = row["nodeID"]

        label = row[label_column] if label_column is not None else None
        nx_graph.add_node(node_id, type="L", label=label)

    for _, row in edges_df.iterrows():
        id_1 = row["id_1"]
        id_2 = row["id_2"]

        if id_1 not in nx_graph:
            nx_graph.add_node(id_1, type=id_1[0], label=None)
        if id_2 not in nx_graph:
            nx_graph.add_node(id_2, type=id_2[0], label=None)

        relation = row["type"]
        nx_graph.add_edge(id_1, id_2, relation=relation)

    return nx_graph
