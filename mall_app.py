import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
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
        st.info("Low Income, High Spending")
    elif cluster == 3:
        st.info("High Income, Low Spending (Target Customers 🎯)")
    else:
        st.info("Average Customers")

    # Insight
    st.markdown("### 💡 Insight")
    st.write("High income but low spending customers are target customers for marketing.")

st.markdown("### 📊 Customer Segments Visualization")

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Predict clusters
clusters = model.predict(X)

# Plot
fig, ax = plt.subplots()

scatter = ax.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=clusters
)

centers = model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')

ax.legend(*scatter.legend_elements(), title="Clusters")

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")

# Show in Streamlit
st.pyplot(fig)
