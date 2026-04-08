import streamlit as st
import joblib
import numpy as np
from datetime import date

# Load weather_models
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

# PREDICT BUTTOn
if st.button("Predict Weather"):

    # Prepare inputs
    temp_input = np.array([[humidity, wind_speed]])
    rain_input = np.array([[humidity, wind_speed, 1]])  # simple placeholder encoding

    # Predictions
    temperature = weather_model.predict(temp_input)[0]
    rain_pred = rain_model.predict(rain_input)[0]
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

    # 7-DAY FORECAST
    st.subheader("📅 7-Day Forecast")

    future_data = []
    for i in range(7):
        future_data.append({
            "Day": f"Day {i+1}",
            "Temperature": round(temperature, 2),
            "Rain %": round(rain_prob * 100, 2)
        })

    st.dataframe(pd.DataFrame(future_data))

    # 🔥 EXTREME WEATHER (IMPORTANT: INSIDE BUTTON)
    st.subheader("⚠️ Extreme Weather Detection")

    if temperature > 35:
        st.error("🔥 Heatwave Warning!")

    elif rain_prob > 0.7:
        st.warning("🌧️ Heavy Rain Expected!")

    elif wind_speed > 40:
        st.warning("🌪️ Storm Alert!")

    else:
        st.success("✅ Weather conditions are normal")

import pandas as pd
import matplotlib.pyplot as plt

st.markdown("---")
st.header("📊 Data Visualization")

# Load dataset
df = pd.read_csv("weatherHistory.csv")

# Convert date
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True, errors='coerce')
df = df.dropna(subset=['Formatted Date'])

# Temperature Trends
st.subheader("🌡️ Temperature Trends Over Time")

temp_trend = df.groupby(df['Formatted Date'].dt.date)['Temperature (C)'].mean()

fig1, ax1 = plt.subplots()
ax1.plot(temp_trend.index[:100], temp_trend.values[:100])
ax1.set_xlabel("Date")
ax1.set_ylabel("Temperature (C)")

st.pyplot(fig1)

# Extract month
df['month'] = df['Formatted Date'].dt.month

st.subheader("🌸 Seasonal Temperature Patterns")

season_temp = df.groupby('month')['Temperature (C)'].mean()

fig2, ax2 = plt.subplots()
ax2.plot(season_temp.index, season_temp.values)
ax2.set_xlabel("Month")
ax2.set_ylabel("Avg Temperature (C)")

st.pyplot(fig2)

st.subheader("🌧️ Rainfall Distribution")

rain_counts = df['Precip Type'].value_counts()

fig3, ax3 = plt.subplots()
ax3.pie(rain_counts, labels=rain_counts.index, autopct='%1.1f%%')

st.pyplot(fig3)
