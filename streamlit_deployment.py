import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  
MODEL_PATH = BASE_DIR / "heart_rf_deploy_pipeline.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

pipeline = load_model()

st.title("Heart Disease Risk – Random Forest Demo")
st.write("Enter patient features and get a predicted heart-disease risk (demo only).")

# ==== 2. Build the input form ====
st.sidebar.header("Patient Features")

# Continuous
age = st.sidebar.slider("Age (years)", 20, 90, 54)

# Sex (encoded: Male=1, Female=0)
sex_label = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_label == "Male" else 0

# Chest pain type (cp) – still codes, but text is clear
cp_options = {
    "Typical angina": 3,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 0,
}
cp_label = st.sidebar.selectbox("Chest pain type (cp)", list(cp_options.keys()))
cp = cp_options[cp_label]

trestbps = st.sidebar.slider("Resting blood pressure (trestbps, mmHg)", 80, 220, 130)
chol = st.sidebar.slider("Serum cholesterol (chol, mg/dL)", 100, 600, 246)

# Fasting blood sugar (fbs: 0/1)
fbs_options = {
    "No (<= 120 mg/dL)": 0,
    "Yes (> 120 mg/dL)": 1,
}
fbs_label = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dL (fbs)", list(fbs_options.keys()))
fbs = fbs_options[fbs_label]

# Resting ECG (restecg)
restecg_options = {
    "Normal": 1,          # mapping may differ in your encoding, but keep consistent
    "ST-T abnormality": 2,
    "LV hypertrophy": 0,
}
restecg_label = st.sidebar.selectbox("Resting ECG result (restecg)", list(restecg_options.keys()))
restecg = restecg_options[restecg_label]

# Max heart rate (thalch – note spelling)
thalch = st.sidebar.slider("Max heart rate achieved (thalch, bpm)", 60, 220, 150)

# Exercise-induced angina (exang)
exang_options = {
    "No": 0,
    "Yes": 1,
}
exang_label = st.sidebar.selectbox("Exercise-induced angina (exang)", list(exang_options.keys()))
exang = exang_options[exang_label]

oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.0, 1.2, step=0.1)

# Slope
slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2,
}
slope_label = st.sidebar.selectbox("Slope of peak exercise ST segment (slope)", list(slope_options.keys()))
slope = slope_options[slope_label]

# Number of major vessels (ca)
ca_options = {
    "0 – no major vessels visible": 0,
    "1 – one vessel": 1,
    "2 – two vessels": 2,
    "3 – three vessels": 3,
    "4 – four vessels": 4,
}
ca_label = st.sidebar.selectbox(
    "Number of major blood vessels (0–4) seen on fluoroscopy (ca)",
    list(ca_options.keys())
)
ca = ca_options[ca_label]

# Thal
thal_options = {
    "Normal perfusion": 1,
    "Fixed defect": 0,
    "Reversible defect": 2,
    "Other / unknown": 3,
}
thal_label = st.sidebar.selectbox("Thalassemia test result (thal)", list(thal_options.keys()))
thal = thal_options[thal_label]

# Make sure keys EXACTLY match training columns
input_dict = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalch,   # important spelling
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}

input_df = pd.DataFrame([input_dict])

st.subheader("Input summary")
st.write(input_df)

# ==== 3. Run prediction ====
if st.button("Predict"):
    proba = pipeline.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)

    label = "Disease" if pred == 1 else "No disease"
    st.markdown(f"### Prediction: **{label}**")
    st.write(f"Estimated probability of disease: **{proba:.3f}**")
    st.info("Educational demo only – not for real medical decisions.")
