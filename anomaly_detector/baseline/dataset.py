from collections import Counter
from dataclasses import dataclass
from itertools import chain, repeat

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from anomaly_detector.data_commons import (
    load_figraph_data,
    merge_df_dict,
    unique_valid_companies,
)
from anomaly_detector.graph.nx_graph import build_networkx_graph


@dataclass(frozen=True)
class TabularDataset:
    train_x: pd.DataFrame
    train_y: pd.DataFrame
    valid_x_dict: dict[str, pd.DataFrame]
    valid_y_dict: dict[str, pd.DataFrame]
    unique_valid_x: pd.DataFrame
    unique_valid_y: pd.DataFrame
    test_x_dict: dict[str, pd.DataFrame]
    test_y_dict: dict[str, pd.DataFrame]

    @property
    def get_valid_years(self) -> list[str]:
        return list(self.valid_x_dict.keys())


def _split_features_and_labels(
    df: pd.DataFrame,
    label_column: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    return df.drop(columns=[label_column]), df[label_column]


def build_tabular_column_transformer(
    companies_df: pd.DataFrame,
    one_hot_columns: list[str],
    drop_columns: list[str],
) -> ColumnTransformer:
    ct = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(sparse_output=False),
                one_hot_columns,
            ),
            ("drop", "drop", drop_columns),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    ct.fit(companies_df)

    return ct


def build_baseline_dataset(
    train_year_start: int,
    train_year_end: int,
    valid_year_start: int,
    valid_year_end: int,
    drop_columns: list[str],
    one_hot_columns: list[str],
    test_year_start: int | None = None,
    test_year_end: int | None = None,
    graph_features: bool = False,
) -> TabularDataset:
    assert train_year_start <= train_year_end, (
        "train_year_start must be less than or equal to train_year_end"
    )
    assert valid_year_start <= valid_year_end, (
        "valid_year_start must be less than or equal to valid_year_end"
    )
    assert valid_year_start > train_year_end, (
        "valid_year_start must be greater than train_year_end"
    )

    train_df_dict, train_edges_dict = load_figraph_data(
        years=range(train_year_start, train_year_end + 1),
    )
    valid_df_dict, valid_edges_dict = load_figraph_data(
        years=range(valid_year_start, valid_year_end + 1),
    )
    test_df_dict, test_edges_dict = load_figraph_data(
        years=range(test_year_start, test_year_end + 1)
        if test_year_start is not None
        else [],
    )

    if graph_features:
        for year, df in train_df_dict.items():
            train_df_dict[year] = add_tabular_graph_features(
                df, train_edges_dict[year])

        for year, df in valid_df_dict.items():
            valid_df_dict[year] = add_tabular_graph_features(
                df, valid_edges_dict[year])

        for year, df in test_df_dict.items():
            test_df_dict[year] = add_tabular_graph_features(
                df, test_edges_dict[year])

    train_df_merged = merge_df_dict(train_df_dict)
    valid_df_merged = merge_df_dict(valid_df_dict)
    test_df_merged = merge_df_dict(test_df_dict) if test_df_dict else None

    unique_valid_df = unique_valid_companies(train_df_merged, valid_df_merged)

    # Build preprocessor
    ct = build_tabular_column_transformer(
        train_df_merged,
        one_hot_columns=one_hot_columns,
        drop_columns=drop_columns,
    )

    # Apply column transformer to datasets
    train_df_merged = ct.transform(train_df_merged)
    valid_df_merged = ct.transform(valid_df_merged)
    unique_valid_df = ct.transform(unique_valid_df)
    if test_df_merged is not None:
        test_df_merged = ct.transform(test_df_merged)

    # Split features and labels
    train_x, train_y = _split_features_and_labels(train_df_merged)
    unique_valid_x, unique_valid_y = _split_features_and_labels(
        unique_valid_df)

    # Preprocess validation data for each year and store in a dictionary
    valid_x_dict, valid_y_dict = {}, {}
    test_x_dict, test_y_dict = {}, {}

    for dset, (year, df) in chain(
        zip(repeat("valid"), valid_df_dict.items()),
        zip(repeat("test"), test_df_dict.items()),
    ):
        # Apply column transformer
        df = ct.transform(df)

        # Split features and labels
        year_x, year_y = _split_features_and_labels(df)

        if dset == "valid":
            valid_x_dict[year] = year_x
            valid_y_dict[year] = year_y
        else:
            test_x_dict[year] = year_x
            test_y_dict[year] = year_y

    return TabularDataset(
        train_x=train_x,
        train_y=train_y,
        valid_x_dict=valid_x_dict,
        valid_y_dict=valid_y_dict,
        unique_valid_x=unique_valid_x,
        unique_valid_y=unique_valid_y,
        test_x_dict=test_x_dict,
        test_y_dict=test_y_dict,
    )


def add_tabular_graph_features(
    companies_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:

    assert len(companies_df["Year"].unique()) == 1, (
        "companies_df should contain data for a single year"
    )
    assert len(edges_df["year"].unique()) == 1, (
        "edges_df should contain data for a single year"
    )

    graph = build_networkx_graph(companies_df, edges_df)

    data_list = companies_df.to_dict(orient="records")

    for i in range(len(data_list)):
        row = data_list[i]

        node_id = row["nodeID"]

        neighbor_types = Counter()
        neighbor_relations = Counter()
        neighbor_relations_with_type = Counter()

        for neighbor, edge in graph.adj[node_id].items():
            neighbor_types[graph.nodes[neighbor]["type"]] += 1
            neighbor_relations[edge["relation"]] += 1
            neighbor_relations_with_type[
                (edge["relation"], graph.nodes[neighbor]["type"])
            ] += 1

        # Node degree
        row["n_nb"] = neighbor_types.total()

        # Count neighbors by type
        row["n_nb_L"] = neighbor_types["L"]
        row["n_nb_H"] = neighbor_types["H"]
        row["n_nb_A"] = neighbor_types["A"]
        row["n_nb_U"] = neighbor_types["U"]
        row["n_nb_O"] = neighbor_types["O"]

        # Count relationship by type
        row["n_aud"] = neighbor_relations["Audit"]
        row["n_rpt"] = neighbor_relations["Related-party transaction"]
        row["n_inv"] = neighbor_relations["Investment"]
        row["n_supply"] = neighbor_relations["Supply chain"]

        # Count relationship by type and neighbor type
        # Audit by A skipped because it is always A
        row["n_inv_H"] = neighbor_relations_with_type[("Investment", "H")]
        row["n_inv_L"] = neighbor_relations_with_type[("Investment", "L")]
        row["n_inv_O"] = neighbor_relations_with_type[("Investment", "O")]
        row["n_inv_U"] = neighbor_relations_with_type[("Investment", "U")]
        row["n_rpt_H"] = neighbor_relations_with_type[
            ("Related-party transaction", "H")
        ]
        row["n_rpt_L"] = neighbor_relations_with_type[
            ("Related-party transaction", "L")
        ]
        row["n_rpt_O"] = neighbor_relations_with_type[
            ("Related-party transaction", "O")
        ]
        row["n_rpt_U"] = neighbor_relations_with_type[
            ("Related-party transaction", "U")
        ]
        row["n_supply_L"] = neighbor_relations_with_type[("Supply chain", "L")]

    df = pd.DataFrame(data_list)

    df["has_listed_nb"] = df["n_nb_L"] > 0
    df["has_human_nb"] = df["n_nb_H"] > 0
    df["has_auditor_nb"] = df["n_nb_A"] > 0
    df["has_unlisted_nb"] = df["n_nb_U"] > 0
    df["has_other_nb"] = df["n_nb_O"] > 0

    return df
