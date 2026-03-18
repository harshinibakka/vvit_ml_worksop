import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title("VVIT Taxi Fare Prediction App (End-to-End ML)")

@st.cache_data
def load_data():
    url = "taxis.csv"
    df = pd.read_csv(url)
    df = df.convert_dtypes()
    st.write(df.head())
    return df

df = load_data()

st.subheader("Dataset Preview")

df = df[['distance', 'fare']].dropna()
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')

x = df[['distance']]
y = df['fare']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Model Performance")
st.write(f"R2 Score: {r2:.2f}")
st.write(f"RMSE: {rmse:.2f}")

st.subheader("Enter Trip Details")

distance = st.number_input(
    "Step 1: Enter Distance (km)",
    min_value=0.0,
    value=5.0
)

passengers = st.number_input(
    "Step 2: Number of Passengers",
    min_value=1,
    value=1
)

hour = st.number_input(
    "Step 3: Hour of Day (0-23)",
    min_value=0,
    max_value=23,
    value=12
)

if st.button("Predict Fare"):
    input_data = np.array([[distance]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Fare: ${prediction[0]:.2f}")
