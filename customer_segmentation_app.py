import streamlit as st
import pickle
import numpy as np

# Load model
with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🛍️ Mall Customer Segmentation App")

st.write("Enter customer details to find their segment")

# User inputs
income = st.number_input("Annual Income (k$)", min_value=0)
spending = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

# Predict
if st.button("Find Customer Segment"):
    data = np.array([[income, spending]])
    cluster = model.predict(data)[0]

    st.success(f"Cluster: {cluster}")

    # Meaning of cluster
    if cluster == 0:
        st.info("Low Income, Low Spending")
    elif cluster == 1:
        st.info("High Income, High Spending (Premium Customers)")
    elif cluster == 2:
        st.info("High Income, Low Spending (Target Customers 🎯)")
    elif cluster == 3:
        st.info("Low Income, High Spending")
    else:
        st.info("Average Customers")

    st.markdown("### 💡 Insight")
    st.write("High income but low spending customers are target customers for marketing.")
