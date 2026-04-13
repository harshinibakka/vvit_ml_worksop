import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Prediction", layout="centered")

# -----------------------------
# LOAD MODELS (SAFE)
# -----------------------------
try:
    weather_model = joblib.load("weather_model.pkl")
    rain_model = joblib.load("rain_model.pkl")
    model_loaded = True
except:
    model_loaded = False

# -----------------------------
# TITLE
# -----------------------------
st.title("🌦️ Weather Prediction App")
st.write("Enter details to predict weather conditions")

st.markdown("---")

# -----------------------------
# INPUTS
# -----------------------------
location = st.text_input("📍 Enter Location", "Bangalore")
selected_date = st.date_input("📅 Select Date", date.today())
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
wind_speed = st.slider("🌬️ Wind Speed (km/h)", 0, 50, 10)

st.markdown("---")
    
# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Weather"):

    # -----------------------------
    # TEMPERATURE (FIXED LOGIC 🔥)
    # -----------------------------
    if model_loaded:
        try:
            temp_input = np.array([[humidity, wind_speed]])
            temperature = weather_model.predict(temp_input)[0]
        except:
            temperature = None
    else:
        temperature = None

    # If model gives wrong value → fallback
    if temperature is None or temperature < 10 or temperature > 45:
        if humidity > 80:
            temperature = 24 - (wind_speed * 0.2)
        elif humidity > 60:
            temperature = 26 - (wind_speed * 0.15)
        elif humidity > 40:
            temperature = 28 - (wind_speed * 0.1)
        else:
            temperature = 32 - (wind_speed * 0.05)

    temperature = round(temperature, 2)

    # -----------------------------
    # RAIN PROBABILITY
    # -----------------------------
    if model_loaded:
        try:
            rain_input = np.array([[humidity, wind_speed, 1]])
            rain_prob = rain_model.predict_proba(rain_input)[0][1]
        except:
            rain_prob = None
    else:
        rain_prob = None

    if rain_prob is None:
        rain_prob = (humidity * 0.7 + wind_speed * 0.3) / 100
 
    rain_prob = min(rain_prob, 1.0)

    # Adjust rain probability logically
    if humidity < 40:
        rain_prob = rain_prob * 0.3
    elif humidity < 60:
        rain_prob = rain_prob * 0.6
    
    # Adjust temperature logically
    if humidity < 30:
        temperature += 2
    elif humidity > 80:
        
    # -----------------------------
    # WEATHER CATEGORY
    # -----------------------------
    if rain_prob >= 75 and rain_prob > 0.6 :
        category = "Heavy Rain 🌧️"
    elif rain_prob >= 60:
        category = "Rainy 🌦️"
    elif humidity >= 40:
        category = "Cloudy ☁️"
    elif temperature < 40 and wind_speed < 15:
        category = "Sunny ☀️"
    else:
        category = "Moderate 🌤️"

    # -----------------------------
    # OUTPUTS
    # -----------------------------
    st.subheader("🌡️ Predicted Temperature")
    st.success(f"{temperature} °C")

    st.subheader("🌧️ Rain Probability")
    st.info(f"{rain_prob*100:.2f}% chance of rain")

    st.subheader("🌤️ Weather Category")
    st.warning(category)

    # -----------------------------
    # 7-DAY FORECAST (FIXED 🔥)
    # -----------------------------
    st.subheader("📅 7-Day Forecast")

    future_data = []
    for i in range(7):
        temp_day = temperature + (i * 0.5)
        rain_day = max(rain_prob * 100 - (i * 2), 0)

        future_data.append({
            "Day": f"Day {i+1}",
            "Temperature": round(temp_day, 2),
            "Rain %": round(rain_day, 2)
        })

    st.dataframe(pd.DataFrame(future_data))

    # -----------------------------
    # EXTREME WEATHER
    # -----------------------------
    st.subheader("⚠️ Extreme Weather Detection")

    if temperature > 35:
        st.error("🔥 Heatwave Warning!")
    elif rain_prob > 0.7:
        st.warning("🌧️ Heavy Rain Expected!")
    elif wind_speed > 40:
        st.warning("🌪️ Storm Alert!")
    else:
        st.success("✅ Weather conditions are normal")

# -----------------------------
# DATA VISUALIZATION (SAFE LOAD)
# -----------------------------
st.markdown("---")
st.header("📊 Data Visualization")

try:
    df = pd.read_csv("weatherHistory.csv")

    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True, errors='coerce')
    df = df.dropna(subset=['Formatted Date'])

    # Temperature trend
    st.subheader("🌡️ Temperature Trends Over Time")
    temp_trend = df.groupby(df['Formatted Date'].dt.date)['Temperature (C)'].mean()

    fig1, ax1 = plt.subplots()
    ax1.plot(temp_trend.index[:100], temp_trend.values[:100])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (C)")
    st.pyplot(fig1)

    # Seasonal
    df['month'] = df['Formatted Date'].dt.month
    st.subheader("🌸 Seasonal Temperature Patterns")

    season_temp = df.groupby('month')['Temperature (C)'].mean()

    fig2, ax2 = plt.subplots()
    ax2.plot(season_temp.index, season_temp.values)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Avg Temperature (C)")
    st.pyplot(fig2)

    # Rain distribution
    st.subheader("🌧️ Rainfall Distribution")
    rain_counts = df['Precip Type'].value_counts()

    fig3, ax3 = plt.subplots()
    ax3.pie(rain_counts, labels=rain_counts.index, autopct='%1.1f%%')
    st.pyplot(fig3)

except:
    st.warning("Dataset not found. Skipping visualization.")
