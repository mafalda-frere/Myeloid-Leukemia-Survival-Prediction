import numpy as np
import pandas as pd
import re


def parse_iscn(iscn: str) -> dict:
    feats = {
        "chromosome_count": np.nan,
        "sex": -1,
        "num_translocations": 0,
        "num_deletions": 0,
        "num_duplications": 0,
        "num_additions": 0,
        "num_monosomies": 0,
        "num_trisomies": 0,
        "abnormal_fraction": 0,
        "is_complex": False,
    }

    sex = {
        "XY": 0,
        "XX": 1,
    }
    if not isinstance(iscn, str):
        return feats

    s = iscn.lower().strip()

    # handle 'complex'
    if "complex" in s:
        feats["is_complex"] = True
        return feats

    # base chromosome count and sex
    base_match = re.match(r"(\d+),([xy]{1,2})", s)
    if base_match:
        feats["chromosome_count"] = int(base_match.group(1))
        feats["sex"] = sex.get(base_match.group(2).upper(), -1)

    # abnormal/normal clone fractions [n]/[m]
    counts = re.findall(r"\[(\d+)\]", s)
    if len(counts) >= 2:
        a, n = map(int, counts[:2])
        feats["abnormal_fraction"] = a / (a + n)

    # abnormality type counters
    feats["num_translocations"] = len(re.findall(r"t\(\d+;\d+\)", s))
    feats["num_deletions"] = len(re.findall(r"del\(\d+\)", s))
    feats["num_duplications"] = len(re.findall(r"dup\(\d+\)", s))
    feats["num_additions"] = len(re.findall(r"add\(\d+\)", s))

    # monosomy/trisomy shorthand (+7, -5, etc.)
    feats["num_monosomies"] = len(re.findall(r"-\d+", s))
    feats["num_trisomies"] = len(re.findall(r"\+\d+", s))

    return feats


def add_cytogenetic_features(
    data: pd.DataFrame, column="CYTOGENETICS"
) -> tuple[pd.DataFrame, list]:
    """
    Apply ISCN parser and merge new features into the given DataFrame.
    """
    ids = data.index
    parsed = data[column].apply(parse_iscn)
    parsed_data = pd.DataFrame(parsed.tolist(), index=data.index)
    return pd.concat([data, parsed_data], axis=1), parsed_data.columns
