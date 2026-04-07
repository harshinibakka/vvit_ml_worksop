import streamlit as st
import joblib
import numpy as np

# Load models
weather_model = joblib.load("weather_model.pkl")
rain_model = joblib.load("rain_model.pkl")

st.set_page_config(page_title="Weather Prediction", layout="centered")

# Title
st.title("🌦️ Weather Prediction App")
st.write("Predict Temperature and Rain Probability")

st.markdown("---")

# Inputs
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)

# Summary input (must match training encoding)
summary_options = [
    "Clear", "Partly Cloudy", "Mostly Cloudy", "Overcast",
    "Rain", "Foggy", "Windy"
]

summary = st.selectbox("Weather Summary", summary_options)

# Simple encoding (same order as training ideally)
summary_mapping = {label: idx for idx, label in enumerate(summary_options)}
summary_encoded = summary_mapping[summary]

st.markdown("---")

# Prediction button
if st.button("Predict"):

    # Temperature prediction
    temp_input = np.array([[humidity, wind_speed]])
    temperature = weather_model.predict(temp_input)[0]

    # Rain prediction
    rain_input = np.array([[humidity, wind_speed, summary_encoded]])
    rain_pred = rain_model.predict(rain_input)[0]

    st.subheader("🌡️ Predicted Temperature:")
    st.success(f"{temperature:.2f} °C")

    st.subheader("🌧️ Rain Prediction:")
    if rain_pred == 1:
        st.error("Rain Expected ☔")
    else:
        st.success("No Rain 🌤️")
