#%%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
# %%
"""
Comprehensive Feature Engineering for Survival Modeling
"""

import numpy as np
import pandas as pd
from tree_based_models import model_selection_using_kfold, get_model

from feature_engineering import (
    one_hot_aggregate,
    add_cytogenetic_features,
    create_molecular_feat,
)

# %%
# Clinical Data
clinical_train = pd.read_csv("data/X_train/clinical_train.csv")
clinical_test = pd.read_csv("data/X_test/clinical_test.csv")

# Molecular Data
molecular_train = pd.read_csv("data/X_train/molecular_train.csv")
molecular_test = pd.read_csv("data/X_test/molecular_test.csv")

target_train = pd.read_csv("data/X_train/target_train.csv")

# Preview the data
clinical_train.head()
# %%
train = pd.concat(
    [
        clinical_train.set_index("ID"),
        target_train.set_index("ID"),
    ],
    axis=1,
)
train = train[~train["OS_YEARS"].isna()]
train.head()


# %%
def feat_engineering(
    data: pd.DataFrame,
    molecular_data: pd.DataFrame,
    fill_not_molecular=False,
) -> tuple[pd.DataFrame, list]:
    """Apply domain-specific feature engineering for DeepSurv."""

    # Identify patients with no molecular data
    ids_not_molecular = [
        pid for pid in data.index.unique() if pid not in molecular_data["ID"].unique()
    ]
    not_molecular = data[data.index.isin(ids_not_molecular)]

    # 1. Molecular Features
    # Aggregates mutation-based numerical features (e.g., count, mean VAF, length)
    data, molecular_feat = create_molecular_feat(
        data=data, molecular_data=molecular_data
    )

    # 2. Cytogenetic Features
    # Adds features describing cytogenetic abnormalities (e.g., complex karyotype)
    data, col_clinical = add_cytogenetic_features(data)

    # 3. One-Hot Encoding of Molecular Categories
    # EFFECT: mutation effect types (e.g., missense, frameshift)
    data, categories = one_hot_aggregate(molecular_data, data, "EFFECT")

    # CHR: chromosome-level distribution of mutations
    data, chromosomes = one_hot_aggregate(
        molecular_data, data, "CHR", fillna_value="no_chr"
    )

    # GENE: gene-level presence/absence indicators
    data, genes = one_hot_aggregate(
        molecular_data, data, "GENE", fillna_value="no_gene"
    )

    # Collect all generated feature names
    new_feats = list(col_clinical) + list(categories) + list(chromosomes) + list(genes)

    # Recompute cytogenetic features for non-molecular patients
    not_molecular, _ = add_cytogenetic_features(not_molecular)

    # Fill missing molecular features with 0 for non-molecular patients
    if fill_not_molecular:
        data = pd.concat([data, not_molecular])
        data[
            list(molecular_feat) + list(categories) + list(chromosomes) + list(genes)
        ] = data[
            list(molecular_feat) + list(categories) + list(chromosomes) + list(genes)
        ].fillna(
            0
        )

    return data, new_feats


# %%
# Apply Feature Engineering
test = clinical_test.set_index("ID").copy()
not_molecular_test = clinical_test[
    clinical_test["ID"].isin(
        [
            id
            for id in clinical_test["ID"].unique()
            if id not in molecular_test["ID"].unique()
        ]
    )
].set_index("ID")

train, feat_train = feat_engineering(
    data=train, molecular_data=molecular_train, fill_not_molecular=True
)
test, feat_test = feat_engineering(
    data=test, molecular_data=molecular_test, fill_not_molecular=True
)

feats = [ft for ft in feat_test if ft in feat_train]
# %%
# Define Model Inputs
target = "OS_YEARS"
features = ["BM_BLAST", "WBC", "HB", "PLT"] + feats
model_type = "cat"  # Parameters and Model based on previous challenges knowledge tree_based_models/initialise_model.py

# %%
# Model Selection with Cross-Validation
model_selection_using_kfold(
    data=train.reset_index(),
    target=target,
    features=features,
    model_type=model_type,
    unique_id="ID",
    plot_ft_importance=True,
    n_splits=6,
    scale=True,
    n_importance=20,
)

# %% Fit model on the whole DataSet

model = get_model(model_type=model_type)
model.fit(train[features], train[target])

# %%
# Generate Predictions and Save Submission
preds = -model.predict(test[features])

submission = pd.Series(preds, index=test.index, name="risk_score")

submission.to_csv("data/cat_feat_engineering_cypt.csv")

# %%
benchmark_ids = set(pd.read_csv("data/benchmark_submission.csv")["ID"])
submission_ids = set(submission.index)

len(benchmark_ids & submission_ids) / len(submission_ids)
# %%
