# Myeloid Leukemia Survival Prediction - ENS Data Challenge

## Overview

This repository contains our submission for the **ENS Data Challenge**:
**“Prédiction de Survie Globale de patients atteints de Leucémie Myéloïde”**
Challenge link: [https://challengedata.ens.fr/challenges/162](https://challengedata.ens.fr/challenges/162)

The objective is to predict a **risk score** for overall survival (OS) in adult patients diagnosed with a subtype of myeloid leukemia. Models are evaluated using the **IPCW C-index**, which measures the concordance between predicted risks and observed survival times under right-censoring.

This project develops and evaluates multiple machine learning and deep learning survival models, from simple baselines to advanced neural survival architectures.

> **Note:** This is a work in progress. The repository structure and codebase are still evolving.

---

## Main Contributions

* A **clean and reproducible modeling pipeline**, integrating preprocessing, feature engineering, and survival model evaluation.
* A set of **baseline IPCW-C-index benchmarks** for classical ML and survival models.
* A **DeepSurv implementation** with custom training loops and IPCW evaluation.
* A systematic **feature engineering approach** combining clinical data, cytogenetic information, and somatic mutation features.
* Contributions to challenge methodology through extensive comparison of linear, tree-based, and deep models.

---

## Data

The challenge dataset includes:

* **Training set:** 3,323 patients
* **Test set:** 1,193 patients

Two main data tables are provided:

1. **Clinical data** (one row per patient)
2. **Molecular data** (one row per detected somatic mutation)

Training labels include:

* **OS_time** - overall survival time
* **OS_event** - censoring indicator (1 = death, 0 = censored)

---

## Methodology & Results

### 1. Exploratory Analysis

* Global statistics & distributions
* Missing values and feature correlations
* Cytogenetic and molecular markers analysis
* Survival curves and event distribution

### 2. Baseline Models

* No feature engineering
* No account for censoring (benchmark)
* **IPCW C-index: 0.6665 ± 0.0091**

### 3. Feature Engineering

* One-hot encoding of clinical categorical variables
* Parsing cytogenetic karyotype strings
* Mutation-based aggregated features
* **IPCW C-index: 0.7075 ± 0.0164**

### 4. Classical Survival Models (sksurv)

* CoxPH
* Random Survival Forests
* **Concordance Index: 0.7160 ± 0.0192**

### 5. Deep Learning - DeepSurv

* PyTorch implementation
* Custom dataloaders and training pipeline
* IPCW-adjusted concordance
* **DeepSurv Test IPCW C-index: 0.7113 ± 0.0107**

---

## Repository Structure

```text
Myeloid-Leukemia-Survival-Prediction/
├── LICENSE
├── README.md
├── requirements.txt        ← dependencies
├── data/                   ← (ignored by Git - must be added locally)
│   ├── X_train/            ← clinical + molecular + targets
│   └── X_test/             ← clinical + molecular
├── notebooks/              ← analysis & presentation workflow
│   ├── explo_surv_data.ipynb
│   ├── base_prediction.ipynb
│   ├── benchmark_challenge.ipynb
│   └── myeloid_survival_prediction.ipynb
├── scripts/                ← runnable experiment scripts
│   ├── survival_tree_pipeline.py
│   ├── deepsurv_pipeline.py
│   └── feature_engineering_experiments.py
└── src/                    ← reusable Python package
    ├── deepsurv/           ← DeepSurv model, training & K-Fold eval
    ├── feature_engineering/← encoding, cytogenetics parser, mutation feats
    └── tree_based_models/  ← CoxPH / RSF pipelines and model selection
```


---

## Future Work
* Better handling of molecular features and biological approach
* Hyperparameter optimization (Optuna / Ray Tune)
* Integration of more advanced architectures TabNet-Surv : intersting while Catboost outerperform other boosting algorithms


