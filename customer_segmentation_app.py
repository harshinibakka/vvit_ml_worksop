#  Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# 📥 Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 👀 View dataset
print(df.head())

# 🎯 Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# =========================
# 🔹 K-MEANS CLUSTERING
# =========================

# Elbow method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure()
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans (k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# =========================
# 📊 2D VISUALIZATION
# =========================

plt.figure()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, cmap='rainbow')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (K-Means)")
plt.show()

# =========================
# 🌳 HIERARCHICAL CLUSTERING
# =========================

Z = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# =========================
# 💡 INSIGHTS
# =========================

print("\n💡 Insights:")
print("High Income + High Spending → Premium Customers")
print("Low Income + Low Spending → Low-value Customers")
print("High Income + Low Spending → Target Customers 🎯")
