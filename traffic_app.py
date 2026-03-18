import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("traffic_model.pkl")

st.title("🚗 Traffic Accident Severity Prediction")

st.write("Enter details to predict accident severity")

# Inputs
lat = st.number_input("Latitude", value=30.0)
lng = st.number_input("Longitude", value=-90.0)
hour = st.slider("Hour (0-23)", 0, 23, 12)
day = st.slider("Day (0=Sunday)", 0, 6, 3)
weather = st.slider("Weather Condition (0-4)", 0, 4, 2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[lat, lng, hour, day, weather]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Accident Severity: {prediction[0]}")
