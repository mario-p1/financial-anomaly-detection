from typing import Literal

from torch_geometric.data import HeteroData
from torch_geometric.transforms import (
    AddMetaPaths,
    RemoveDuplicatedEdges,
    RemoveSelfLoops,
)

L_U_L_METAPATHS = [
    [("L", "Investment", "U"), ("U", "Investment", "L")],
    [("L", "Investment", "U"), ("U", "Related-party transaction", "L")],
    [("L", "Related-party transaction", "U"), ("U", "Investment", "L")],
    [("L", "Related-party transaction", "U"), ("U", "Related-party transaction", "L")],
]
L_H_L_METAPATHS = [
    [("L", "Investment", "H"), ("H", "Investment", "L")],
    [("L", "Investment", "H"), ("H", "Related-party transaction", "L")],
    [("L", "Related-party transaction", "H"), ("H", "Investment", "L")],
    [("L", "Related-party transaction", "H"), ("H", "Related-party transaction", "L")],
]
L_O_L_METAPATHS = [
    [("L", "Investment", "O"), ("O", "Investment", "L")],
    [("L", "Investment", "O"), ("O", "Related-party transaction", "L")],
    [("L", "Related-party transaction", "O"), ("O", "Investment", "L")],
    [("L", "Related-party transaction", "O"), ("O", "Related-party transaction", "L")],
]
ALL_METAPATHS = [*L_U_L_METAPATHS, *L_H_L_METAPATHS, *L_O_L_METAPATHS]


def build_metapath_graph(
    graph: HeteroData,
    metapaths: list[list[tuple[str, str, str]]] = ALL_METAPATHS,
    keep_same_node_type: bool = True,
    reduce: Literal["sum", "mean", "max"] | None = "sum",
) -> HeteroData:
    metapath_graph = AddMetaPaths(
        metapaths=metapaths,
        drop_orig_edge_types=True,
        drop_unconnected_node_types=True,
        keep_same_node_type=keep_same_node_type,
    )(graph)

    metapath_graph = RemoveSelfLoops()(metapath_graph)

    if reduce is not None:
        metapath_graph = RemoveDuplicatedEdges(reduce="sum")(metapath_graph)

    return metapath_graph
