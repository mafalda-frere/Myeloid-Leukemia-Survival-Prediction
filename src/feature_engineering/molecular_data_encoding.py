import pandas as pd
from typing import Tuple


def create_molecular_feat(
    data: pd.DataFrame, molecular_data: pd.DataFrame
) -> Tuple[pd.DataFrame, list]:

    molecular_data["LENGTH"] = molecular_data["END"] - molecular_data["START"]

    tmp = molecular_data.groupby("ID").size().to_frame("Nmut")
    length = molecular_data.groupby("ID")["LENGTH"].sum()
    vaf = molecular_data.groupby("ID")["VAF"].sum()

    data = data.merge(tmp, left_index=True, right_index=True, how="inner")
    data = data.merge(vaf, left_index=True, right_index=True, how="inner")
    data = data.merge(length, left_index=True, right_index=True, how="inner")

    return data, ["LENGTH", "Nmut", "VAF"]
