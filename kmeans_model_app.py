import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Title
# ----------------------------
st.title("🛍️ Mall Customer Clustering App")
st.write("Enter customer details and visualize customer clusters.")

# ----------------------------
# Input: Annual Income and Spending Score
# ----------------------------
income = st.number_input("Enter Annual Income (k$)", min_value=0, max_value=200, value=60)
spending = st.slider("Select Spending Score (1-100)", 1, 100, value=50)

# ----------------------------
# Example Dataset (simulate mall customers)
# ----------------------------
np.random.seed(42)
income_data = np.random.randint(15, 140, 200)
spending_data = np.random.randint(1, 100, 200)
df = pd.DataFrame({'Income': income_data, 'Spending Score': spending_data})

# Add current user input as a new row
df_user = pd.DataFrame({'Income': [income], 'Spending Score': [spending]})
df = pd.concat([df, df_user], ignore_index=True)

# ----------------------------
# Scale Data
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ----------------------------
# K-Means Clustering
# ----------------------------
n_clusters = st.sidebar.slider("Select number of clusters (K-Means)", 2, 10, 5)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# ----------------------------
# Visualization
# ----------------------------
st.subheader("📊 Customer Clusters (2D)")

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(df['Income'], df['Spending Score'], c=df['Cluster'], cmap='Set1', s=50)
ax.set_xlabel("Income (k$)")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation (K-Means)")
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)

# ----------------------------
# Show cluster of user input
# ----------------------------
user_label = df.iloc[-1]['Cluster']
st.success(f"Your input belongs to **Cluster {user_label}**")

# Optional: Show cluster characteristics
cluster_info = df.groupby('Cluster')[['Income','Spending Score']].mean()
st.write("Cluster centroids (average Income & Spending Score):")
st.dataframe(cluster_info)
