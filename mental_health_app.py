import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("mental_health_model.pkl")
columns = joblib.load("columns.pkl")

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
# PREDICTION
# -----------------------------
if st.button("Predict"):

    input_dict = {
        "Age": age,
        "Gender": gender,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "remote_work": remote_work,
        "benefits": benefits,
        "care_options": care_options
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Encoding
    input_df = pd.get_dummies(input_df)

    # Match columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # -----------------------------
    # PROBABILITY BASED PREDICTION
    # -----------------------------
   proba = model.predict_proba(input_df)[0]

   # proba[0] = High Risk probability
   # proba[1] = Low Risk probability

   if proba[0] >= 0.6:
       st.error("🔴 High Risk of Mental Health Issues")
       st.write("👉 Suggestions:")
       st.write("- Improve work-life balance")
       st.write("- Seek professional help")
       st.write("- Talk to HR / support system")
       
   elif proba[0] >= 0.3:
       st.warning("🟡 Medium Risk of Mental Health Issues")
       st.write("👉 Suggestions:")
       st.write("- Take short breaks")
       st.write("- Maintain work-life balance")
       st.write("- Talk to someone you trust")
       
   else:
       st.success("🟢 Low Risk of Mental Health Issues")
       st.write("👉 Keep maintaining a healthy lifestyle 😊")
