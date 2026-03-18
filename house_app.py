import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("house_model.pkl", "rb"))

st.set_page_config(page_title="House Price Prediction")

st.title("🏠 House Price Prediction App")

st.write("Enter details:")

longitude = st.number_input("Longitude", value=-122.0)
latitude = st.number_input("Latitude", value=37.0)
housing_median_age = st.number_input("House Age", value=10)
total_rooms = st.number_input("Total Rooms", value=2000)
total_bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=3.0)

if st.button("Predict"):
    features = np.array([[longitude, latitude, housing_median_age,
                          total_rooms, total_bedrooms,
                          population, households, median_income]])

    prediction = model.predict(features)

    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
