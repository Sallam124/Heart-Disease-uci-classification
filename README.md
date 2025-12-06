# Heart Disease Prediction & Imbalance Handling

This project explores machine learning techniques to predict heart disease risk, specifically addressing the challenge of **severe class imbalance** in healthcare datasets. The primary goal was to maximize **Recall** to ensure potential heart disease cases are not missed.

## Project Overview

- **Problem:** Predicting risk of heart disease (e.g., 10-year CHD risk).
- **Challenge:** Datasets are often highly imbalanced (e.g., ~15% positive cases), causing standard models to bias towards the healthy class.
- **Solution:** Implemented and compared strategies including **SMOTE**, **Risk Clustering**, and **Threshold Tuning**.

## Key Techniques & Results

1.  **Preprocessing:** Robust cleaning, imputing missing values, and feature scaling.
2.  **Imbalance Handling:**
    *   **Class Weighting:** Adjusted loss functions for models like Logistic Regression.
    *   **SMOTE:** Generated synthetic minority samples to balance training.
    *   **Unsupervised Clustering:** Segmented patients into risk groups significantly aiding prediction.
3.  **Models:**
    *   Logistic Regression & Random Forest (Baseline).
    *   Artificial Neural Networks (ANN) with regularization.
4.  **Threshold Tuning:** Optimized decision thresholds to boost Recall from **~2% (Baseline)** to **~80%**, drastically reducing false negatives.

## Tech Stack

- **Python** (Pandas, NumPy)
- **Scikit-learn** (Modeling, Preprocessing)
- **Imbalanced-learn** (SMOTE, Pipeline)
- **TensorFlow/Keras** (ANN)
