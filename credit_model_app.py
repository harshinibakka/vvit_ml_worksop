import streamlit as st
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open("credit_model.pkl(1)", "rb"))

st.title("💳 AI Credit Score Classifier")

# User Inputs based on Feature Engineering
income = st.number_input("Annual Income")
debt = st.number_input("Outstanding Debt")
delayed_pay = st.number_input("Number of Delayed Payments")
interest = st.number_input("Interest Rate")
utilization = st.slider("Credit Utilization Ratio", 0, 100, 30)

# Re-creating the extra features for the model input
debt_ratio = debt / (income + 1)
delay_impact = delayed_pay * interest

# Prepare features (Must match the exact order of X.columns)
# Example: [Income, Salary, Bank, Card, Interest, Loans, DelayDays, DelayNum, Debt, Util, Ratio, Impact]
if st.button("Predict Credit Class"):
    # Note: Using placeholder zeros for columns we didn't use as inputs for simplicity
    input_data = np.array([[income, income/12, 2, 3, interest, 1, 5, delayed_pay, debt, utilization, debt_ratio, delay_impact]])
    
    prediction = model.predict(input_data)[0]
    
    # Map back to labels
    classes = {0: "Poor", 1: "Standard", 2: "Good"}
    result = classes[prediction]
    
    if result == "Good":
        st.success(f"Credit Score Category: {result}")
        st.balloons()
    elif result == "Standard":
        st.warning(f"Credit Score Category: {result}")
    else:
        st.error(f"Credit Score Category: {result}")
