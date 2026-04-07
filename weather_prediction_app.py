import streamlit as st
import pickle

st.markdown("---")
st.subheader("🌦️ Weather Prediction")

# Load model
model = pickle.load(open("weather_model.pkl", "rb"))

# Inputs
humidity = st.slider("Humidity", 0, 100)
wind_speed = st.slider("Wind Speed (km/h)", 0, 100)

# Predict
if st.button("Predict Weather"):
    result = model.predict([[humidity, wind_speed]])
    st.success(f"🌡️ Predicted Temperature: {result[0]:.2f} °C")
