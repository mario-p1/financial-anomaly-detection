from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from anomaly_detector.utils import base_data_path, base_project_path


def load_figraph_data(
    years: Iterable[int],
    data_dir: Path = base_data_path / "figraph_raw",
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    tabular_dfs = {}
    for year in years:
        df = pd.read_csv(data_dir / f"ListedCompanyFeatures772_{year}.csv")
        df["Year"] = year
        tabular_dfs[year] = df

    edge_dfs = {}
    for year in years:
        df = pd.read_csv(data_dir / f"edges{year}.csv", header=None)
        df.columns = ["id_1", "id_2", "type"]
        df["year"] = year
        edge_dfs[year] = df

    return tabular_dfs, edge_dfs


def merge_df_dict(df_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    merged_df = pd.concat(df_dict.values(), ignore_index=True)
    return merged_df


def decorate_edges_with_node_types(edge_df: pd.DataFrame) -> pd.DataFrame:
    decorated_df = edge_df.copy()
    decorated_df["node_type_1"] = decorated_df["id_1"].str[0]
    decorated_df["node_type_2"] = decorated_df["id_2"].str[0]

    return decorated_df


def order_edge_nodes(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Order the nodes in each edge such that the node
    with a type starting with "L" is always in the first.
    """
    ordered_df = edge_df.copy()

    # Swap the nodes in the edge if the first node is not of type "L"
    swap = edge_df["id_1"].str[0] != "L"
    ordered_df.loc[swap, ["id_1", "id_2"]] = ordered_df.loc[
        swap,
        ["id_2", "id_1"],
    ].to_numpy()

    # Swap the nodes in the edge if both nodes are of type "L"
    # and the first node is greater than the second node
    swap_l_id = (
        (ordered_df["id_1"].str[0] == "L")
        & (ordered_df["id_2"].str[0] == "L")
        & (ordered_df["id_1"] > ordered_df["id_2"])
    )
    ordered_df.loc[swap_l_id, ["id_1", "id_2"]] = ordered_df.loc[
        swap_l_id,
        ["id_2", "id_1"],
    ].to_numpy()

    # Reapply node type decoration after swapping
    if "node_type_1" in ordered_df.columns:
        ordered_df = decorate_edges_with_node_types(ordered_df)

    return ordered_df


def unique_valid_companies(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> pd.DataFrame:
    train_companies = set(train_df["nodeID"].unique())

    return valid_df[~valid_df["nodeID"].isin(train_companies)].copy()


def load_figraph_columns_df(
    target_column: str,
    csv_path: Path = base_project_path / "figraph_tabular_columns.csv",
) -> pd.DataFrame:
    columns_df = pd.read_csv(csv_path)

    # Extract column groups
    columns_df["column_group"] = columns_df["column"].str.extract(
        r"^([ABCD])(?=\w{2}\d{7}$)",
        expand=False,
    )
    columns_df["column_group"] = columns_df["column_group"].fillna(
        columns_df["column"].str.extract(r"^(F)(?=\d{6}\w?$)", expand=False),
    )
    columns_df["column_group"] = columns_df["column_group"].fillna("Other")

    columns_df.loc[lambda x: x["column"] ==
                   target_column, "column_group"] = "TARGET"

    # Extract column subgroups
    columns_df["column_subgroup"] = columns_df["column"].str.extract(
        r"^[ABCD](\w{2})\d{7}",
        expand=False,
    )
    return columns_df


def find_leafs_edges_df(edges_df: pd.DataFrame) -> set[str]:
    node_counts = pd.concat(
        [edges_df["id_1"], edges_df["id_2"]]).value_counts()

    leaf_nodes = node_counts[node_counts == 1].index.to_list()

    return set(leaf_nodes)


def remove_nodes_in_dataframes(
    tabular_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    nodes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_tabular_df = tabular_df[~tabular_df["nodeID"].isin(nodes)].reset_index(
        drop=True,
    )
    filtered_edges_df = edges_df[
        ~edges_df["id_1"].isin(nodes) & ~edges_df["id_2"].isin(nodes)
    ].reset_index(drop=True)

    return filtered_tabular_df, filtered_edges_df
