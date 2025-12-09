# Heart Disease Risk Prediction (Educational Project)

This project is an **educational demo** that predicts the risk of heart disease from basic clinical information (age, blood pressure, cholesterol, etc.).  

It has two faces:

- A **simple web app** (Streamlit) where you move sliders and get a risk prediction.
- A set of **notebooks** that show the full machine-learning pipeline, including how to handle imbalanced medical data.

> ⚠️ **Important:** This is **not** a medical device. It must **not** be used for real clinical decisions.

---

## 1. What this project does (plain English)

- You enter patient-style information (age, sex, blood pressure, cholesterol, etc.).
- A trained **Random Forest** model estimates the probability that this person has heart disease.
- Behind the scenes, the project shows:
  - Why a **heavily imbalanced dataset** can give fake-good accuracy but miss real patients.
  - How switching to a **better balanced dataset** + proper evaluation leads to much more trustworthy results.
  - How to add a small **unsupervised step (K-Means clustering)** as a risk feature.

The code and report are written as coursework for a machine learning module.

---

## 2. Quick start – run the demo app

### Requirements

- Python 3.9+ (3.10+ recommended)
- `pip` installed

### Install dependencies

From the project root (`ML-Project/`):

```bash
pip install -r requirements.txt
