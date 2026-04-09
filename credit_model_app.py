import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("credit_model.pkl(1)")

st.title("💳 Credit Score Predictor")

st.write("Enter your financial details:")

# Inputs
income = st.number_input("Annual Income")
debt = st.number_input("Outstanding Debt")
payment_history = st.slider("Payment History Score", 0, 100)

# Feature engineering
debt_ratio = debt / income if income != 0 else 0

# Prediction button
if st.button("Predict"):
    features = np.array([[income, debt, payment_history, debt_ratio]])
    
    prediction = model.predict(features)

    if prediction[0] == 0:
        result = "Poor"
    elif prediction[0] == 1:
        result = "Standard"
    else:
        result = "Good"

    st.success(f"Your Credit Score Category: {result}")
