import streamlit as st
import joblib
import numpy as np
from datetime import date

# Load models
weather_model = joblib.load("weather_model.pkl")
rain_model = joblib.load("rain_model.pkl")

st.set_page_config(page_title="Weather Prediction", layout="centered")

# Title
st.title("🌦️ Weather Prediction App")
st.write("Enter details to predict weather conditions")

st.markdown("---")

# INPUTS
location = st.text_input("📍 Enter Location", "Bangalore")

selected_date = st.date_input("📅 Select Date", date.today())

humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
wind_speed = st.slider("🌬️ Wind Speed (km/h)", 0, 50, 10)

st.markdown("---")

# PREDICT BUTTON
if st.button("Predict Weather"):

    # Prepare inputs
    temp_input = np.array([[humidity, wind_speed]])
    rain_input = np.array([[humidity, wind_speed, 1]])  # simple placeholder encoding

    # Predictions
    temperature = weather_model.predict(temp_input)[0]
    rain_pred = rain_model.predict(rain_input)[0]

    # Convert rain to probability (approx)
    rain_prob = rain_model.predict_proba(rain_input)[0][1]

    # Weather category logic (simple rule-based)
    if rain_pred == 1:
        category = "Rainy ☔"
    elif humidity > 70:
        category = "Cloudy ☁️"
    elif temperature > 30:
        category = "Sunny ☀️"
    else:
        category = "Moderate 🌤️"

    # OUTPUTS
    st.subheader("🌡️ Predicted Temperature")
    st.success(f"{temperature:.2f} °C")

    st.subheader("🌧️ Rain Probability")
    st.info(f"{rain_prob*100:.2f}% chance of rain")

    st.subheader("🌤️ Weather Category")
    st.warning(category)
