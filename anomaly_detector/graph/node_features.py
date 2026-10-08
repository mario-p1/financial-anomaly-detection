import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


def build_node_features_preprocessor(
    companies_df: pd.DataFrame,
    one_hot_columns: list[str],
    drop_columns: list[str],
) -> ColumnTransformer:
    numerical = companies_df.columns.difference(one_hot_columns + drop_columns)
    ct = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(sparse_output=False),
                one_hot_columns,
            ),
            (
                "numerical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        # (
                        #     "scaler",
                        #     RobustScaler(),
                        # ),
                        (
                            "scaler",
                            QuantileTransformer(
                                output_distribution="normal", random_state=42
                            ),
                        ),
                    ],
                ),
                numerical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    ct.fit(companies_df)

    return ct
