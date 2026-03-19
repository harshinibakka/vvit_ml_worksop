import streamlit as st
import joblib
import pandas as pd

# Load model and columns
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

    # Keep values as strings (IMPORTANT)
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

    # Apply encoding same as training
    input_df = pd.get_dummies(input_df)

    # Match columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 0:
        st.error("⚠️ High Risk of Mental Health Issues")
        st.write("👉 Suggestions:")
        st.write("- Improve work-life balance")
        st.write("- Seek professional help")
        st.write("- Talk to HR / support system")
    else:
        st.success("✅ Low Risk of Mental Health Issues")
        st.write("👉 Keep maintaining a healthy lifestyle 😊")
