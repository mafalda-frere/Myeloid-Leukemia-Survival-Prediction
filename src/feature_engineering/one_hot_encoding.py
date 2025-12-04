import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple
import re


def one_hot_aggregate(
    molecular_data: pd.DataFrame,
    data: pd.DataFrame,
    column: str,
    fillna_value: str | None = None,
) -> Tuple[pd.DataFrame, list]:
    """
    One-hot encode a categorical column in molecular_data, aggregate by 'ID', and merge with data.
    """
    molecular_data = molecular_data.copy()
    # Fill missing values if requested
    if fillna_value is not None:
        molecular_data = molecular_data.copy()
        molecular_data[column] = molecular_data[column].fillna(fillna_value)

    # Fit encoder
    oht = OneHotEncoder(sparse_output=False)
    encoded = oht.fit_transform(molecular_data[[column]])
    categories = oht.categories_[0]

    # Combine encoded columns with original molecular_data
    encoded_molecular_data = pd.DataFrame(
        encoded, columns=categories, index=molecular_data.index
    )
    molecular_data_encoded = pd.concat(
        [molecular_data[["ID"]], encoded_molecular_data], axis=1
    )

    # Aggregate all at once
    aggregated = molecular_data_encoded.groupby("ID").sum()
    # Merge once (vectorized, not in a loop)
    return (
        data.merge(aggregated, left_index=True, right_index=True, how="inner"),
        categories,
    )
