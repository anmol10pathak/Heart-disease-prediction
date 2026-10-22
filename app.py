import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="HeartCare | Risk Prediction",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM UI DESIGN ----------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b1220 0%, #111c30 100%); color: #f1f5f9; }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label { color: #f1f5f9 !important; }
    .hero { background: linear-gradient(120deg, #162b49, #173b52); padding: 32px; border: 1px solid #294660; border-radius: 22px; margin-bottom: 25px; }
    .hero-title { font-size: 34px; font-weight: 800; color: #ffffff; margin-bottom: 8px; }
    .hero-subtitle { color: #b9c9dc; font-size: 16px; line-height: 1.7; }
    .section-title { font-size: 21px; font-weight: 700; color: #7dd3fc; margin: 14px 0 18px 0; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(20, 35, 56, 0.78); border: 1px solid #293e59; border-radius: 18px; padding: 12px; }
    div[data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; font-weight: 500; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background-color: #111d30; border-color: #344862; border-radius: 10px; }
    div[data-baseweb="select"] *, div[data-baseweb="input"] input { color: #f8fafc !important; }
    div[data-testid="stSlider"] { padding-top: 8px; }
    div.stButton > button { width: 100%; height: 55px; border-radius: 12px; border: none; background: linear-gradient(90deg, #0284c7, #06b6d4); color: white; font-size: 18px; font-weight: 700; transition: 0.2s ease; }
    div.stButton > button:hover { background: linear-gradient(90deg, #0369a1, #0891b2); border: none; color: white; transform: translateY(-2px); }
    .footer { text-align: center; color: #94a3b8; font-size: 13px; margin-top: 35px; padding-top: 18px; border-top: 1px solid #29384e; }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SAVED FILES ----------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="hero-title">💙 HeartCare</div>
    <div class="hero-subtitle">
        HEART HEALTH RISK PREDICTION DASHBOARD<br>
        Enter the health measurements below to generate a
        model-based heart disease risk prediction.
    </div>
</div>
""", unsafe_allow_html=True)
st.caption("🛡️ Personal Health Assessment Interface · Educational Use Only")
st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", 18, 100, 40)
    with col2:
        sex = st.selectbox("Sex", ["M", "F"])
    with col3:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

st.markdown('<div class="section-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    with col2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    with col3:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    col1, col2, col3 = st.columns(3)
    with col1:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    with col2:
        max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    with col3:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

st.markdown('<div class="section-title">📈 Exercise & ECG Indicators</div>', unsafe_allow_html=True)
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    with col2:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("")
_, button_col, _ = st.columns([1, 2, 1])
with button_col:
    predict_clicked = st.button("🔍 Predict Heart Disease Risk")

if predict_clicked:
    raw_input = {
        "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "MaxHR": max_hr, "Oldpeak": oldpeak,
        "Sex_" + sex: 1, "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1, "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.markdown("---")
    st.markdown('<div class="section-title">📋 Prediction Results</div>', unsafe_allow_html=True)
    if prediction == 1:
        st.error("⚠️ The model indicates a higher likelihood of heart disease. This is not a diagnosis. Consult a qualified healthcare professional.")
    else:
        st.success("✅ The model indicates a lower likelihood of heart disease. This does not rule out heart disease or replace medical evaluation.")

st.markdown("""
<div class="footer">
    HEARTCARE · HEART DISEASE PREDICTION PROJECT<br>
    Educational and demonstration purposes only. Not a medical diagnostic tool.
</div>
""", unsafe_allow_html=True)
