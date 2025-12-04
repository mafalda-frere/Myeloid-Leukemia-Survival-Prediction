#%%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
# %%
"""
DeepSurv Pipeline for Survival Prediction
"""

import pandas as pd
import matplotlib.pyplot as plt
from deepsurv import model_selection_using_kfold_deepsurv, DeepSurv, train_deepsurv
from feature_engineering import (
    one_hot_aggregate,
    add_cytogenetic_features,
    create_molecular_feat,
)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# %%
# Load Data
clinical_train = pd.read_csv("data/X_train/clinical_train.csv")
clinical_test = pd.read_csv("data/X_test/clinical_test.csv")

molecular_train = pd.read_csv("data/X_train/molecular_train.csv")
molecular_test = pd.read_csv("data/X_test/molecular_test.csv")

target_train = pd.read_csv("data/X_train/target_train.csv")


# Merge training data
train = pd.concat(
    [
        clinical_train.set_index("ID"),
        target_train.set_index("ID"),
    ],
    axis=1,
)

# Drop rows with missing survival targets
train = train[~train["OS_YEARS"].isna()]
train.head()


# %%
def feat_engineering(
    data: pd.DataFrame,
    molecular_data: pd.DataFrame,
    fill_not_molecular=False,
) -> tuple[pd.DataFrame, list]:

    # Identify patients with no molecular data those are dropped from the dataset
    ids_not_molecular = [
        id for id in data.index.unique() if id not in molecular_data["ID"].unique()
    ]
    not_molecular = data[data.index.isin(ids_not_molecular)]

    # Add cytogenetic and molecular features
    data, col_clinical = add_cytogenetic_features(data)
    data, categories = one_hot_aggregate(molecular_data, data, "EFFECT")
    data, molecular_feat = create_molecular_feat(
        data=data, molecular_data=molecular_data
    )

    new_feats = list(col_clinical) + list(categories) + list(molecular_feat)

    # Add cytogenetic features for non-molecular subset
    not_molecular, col_clinical = add_cytogenetic_features(not_molecular)

    # Optionally fill and merge non-molecular patients
    if fill_not_molecular:
        data = pd.concat([data, not_molecular])
        data[list(categories) + list(molecular_feat)] = data[
            list(categories) + list(molecular_feat)
        ].fillna(0)

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

# Align train/test feature sets
feats = [ft for ft in feat_test if ft in feat_train]

# %%
# Define Feature Columns
target = "OS_YEARS"
status = "OS_STATUS"
features = ["BM_BLAST", "WBC", "HB", "PLT"] + feats

# DeepSurv hyperparameters
hidden_layers_sizes = [64, 32]
dropout = 0.4
n_splits = 6
n_epochs = 50
lr = 1e-4
weight_decay = 1e-4

# %%
# Model Selection with Cross-Validation
results = model_selection_using_kfold_deepsurv(
    data=train.reset_index(),
    features=features,
    target=target,
    status=status,
    hidden_layers_sizes=hidden_layers_sizes,
    dropout=dropout,
    n_splits=n_splits,
    n_epochs=n_epochs,
    lr=lr,
    weight_decay=weight_decay,
    unique_id="ID",
)

# %%
# Data Preprocessing (Imputation + Scaling)
X_train = train[features]
X_test = test[features]
t_train = train[target]
e_train = train[status]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Train Final DeepSurv Model
n_in = X_train.shape[1]
model = DeepSurv(
    n_in=n_in,
    hidden_layers_sizes=hidden_layers_sizes,
    dropout=dropout,
)

# Train model on full training set
model = train_deepsurv(
    model=model,
    x_train=X_train,
    e_train=e_train.values,
    t_train=t_train.values,
    n_epochs=n_epochs,
    lr=lr,
    weight_decay=weight_decay,
    device="cpu",
    verbose=True,
)
# %%
# Generate Predictions and Save Submission
preds = model.predict(X_test)

submission = pd.Series(preds, index=test.index, name="risk_score")
submission.to_csv("data/deep_surv.csv")

# %%
