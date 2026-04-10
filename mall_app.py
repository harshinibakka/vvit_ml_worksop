import streamlit as st
import numpy as np
import pickle
from sklearn.cluster import KMeans  # IMPORTANT

# Title
st.title("🛍️ Mall Customer Clustering App")

st.write("Enter customer details to find their cluster")

# Load model
with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

# Inputs
income = st.number_input("Annual Income (k$)", min_value=0)
spending = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

# Button
if st.button("Find Customer Cluster"):

    data = np.array([[income, spending]])
    cluster = model.predict(data)[0]

    # Output
    st.success(f"Cluster: {cluster}")

    # Meaning
    if cluster == 0:
        st.info("Low Income, Low Spending")
    elif cluster == 1:
        st.info("High Income, High Spending (Premium Customers 💎)")
    elif cluster == 2:
        st.info("High Income, Low Spending (Target Customers 🎯)")
    elif cluster == 3:
        st.info("Low Income, High Spending")
    else:
        st.info("Average Customers")

    # Insight
    st.markdown("### 💡 Insight")
    st.write("High income but low spending customers are target customers for marketing.")
