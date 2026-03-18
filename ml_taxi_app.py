import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


st.title("PragyanAI Taxi Fare Prediction App (End-to-End ML)")
st.cache_data


def load_data():
    url = "taxis.csv"
    df = pd.read_csv(url)
    df = df.convert_dtypes()
    st.write(df.head())   # instead of st.dataframe(df)
    return df


df = load_data()

st.subheader("PragyanAI Dataset Preview")


# Splitting data (same flow they will continue in class)
X = df[['distance']]
y = df['fare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("R2 Score:", r2_score(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))


# Plot
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred)
plt.xlabel("Distance")
plt.ylabel("Fare")

st.pyplot(plt)
