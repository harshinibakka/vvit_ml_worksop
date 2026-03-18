import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("mental_health_model.pkl")

st.set_page_config(page_title="Mental Health Prediction", layout="centered")

st.title("🧠 Mental Health Risk Prediction")
st.write("Fill in the details to check mental health risk")

# -----------------------------
# INPUT FIELDS
# -----------------------------

age = st.slider("Age", 18, 60, 25)

gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History", ["Yes", "No"])
work_interfere = st.selectbox("Work Interference", ["Never", "Rarely", "Sometimes", "Often"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
benefits = st.selectbox("Company Benefits", ["Yes", "No"])
care_options = st.selectbox("Care Options Available", ["Yes", "No"])

# -----------------------------
# ENCODE INPUT (IMPORTANT)
# -----------------------------

# Convert inputs to numbers (same as training)
gender = 1 if gender == "Male" else 0
family_history = 1 if family_history == "Yes" else 0
remote_work = 1 if remote_work == "Yes" else 0
benefits = 1 if benefits == "Yes" else 0
care_options = 1 if care_options == "Yes" else 0

# Work interference encoding
work_map = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3
}
work_interfere = work_map[work_interfere]

# Stress score (feature engineering same as training)
stress_score = 1 if work_interfere >= 2 else 0

# -----------------------------
# FINAL INPUT ARRAY
# -----------------------------

input_data = np.array([[age, gender, family_history, work_interfere,
                        remote_work, benefits, care_options, stress_score]])

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Mental Health Issues")
        st.write("👉 Suggestions:")
        st.write("- Improve work-life balance")
        st.write("- Seek professional help")
        st.write("- Talk to HR / support system")
    else:
        st.success("✅ Low Risk of Mental Health Issues")
        st.write("👉 Keep maintaining a healthy lifestyle 😊")
