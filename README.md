# Heart Disease Risk Prediction

This project builds a machine learning pipeline to predict **heart disease** from basic clinical features (age, chest pain type, blood pressure, cholesterol, etc.).

The main goal:  
Build a model that can detect patients with heart disease reliably, while showing how **data quality and class balance** can completely change model behaviour.

---

## Why there are 2 datasets

### Dataset 1 – Large but badly imbalanced
- File: `Dataset_1/dataset/heart_disease.csv`
- ~4,200 samples with **~15% positive (disease)**.
- Problem:
  - Models (Logistic Regression, Random Forest, even ANN) could reach **high accuracy**, but:
    - Recall for the disease class was extremely low in many setups (e.g. a few %).
  - Even after:
    - Class weights  
    - SMOTE  
    - K-Means risk clustering  
    - Threshold tuning  
  - You either:
    - Miss most real patients, or  
    - Flag too many healthy people as diseased.

Result: Dataset 1 is kept to **document and demonstrate** the imbalance problem and the failed attempts. It is *not* used for final results or deployment.

---

### Dataset 2 – Smaller but reasonably balanced (UCI-style)
- File: `Dataset_2/dataset/heart_disease_uci.csv`
- ~900 patients, standard UCI-style heart disease data.
- Target:
  - Original `num` ∈ {0,1,2,3,4}  
  - Converted to binary:
    - `0`  → no disease  
    - `1–4` → disease
- Class balance is much better, so metrics like recall, F1, and ROC-AUC actually reflect how well the model detects heart disease.

On this dataset the final pipeline is built:

- Clean preprocessing (imputation, encoding, scaling)
- Logistic Regression vs Random Forest comparison
- Simple unsupervised step (K-Means) used as a **risk_cluster** feature
- Cross-validation and basic hyperparameter tuning
- Final **Random Forest pipeline** saved as:
  - `heart_rf_deploy_pipeline.joblib`

All final results in the report and the Streamlit demo are based on **Dataset 2**.

---

## How to run

From the project root (`ML-Project/`):

### 1. Install dependencies

```bash
pip install -r requirements.txt
