import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_customers = 500
data = {
    'Recency': np.random.randint(1, 365, num_customers),  # Days since last purchase
    'Frequency': np.random.poisson(lam=5, size=num_customers),  # Number of purchases
    'MonetaryValue': np.random.exponential(scale=100, size=num_customers), # Average purchase value
    'Churn': np.random.binomial(1, 0.2, num_customers) # 0: Not churned, 1: Churned
}
df = pd.DataFrame(data)
# --- 2. Data Preprocessing ---
# Scale the features for KMeans clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Recency', 'Frequency', 'MonetaryValue']])
# --- 3. Customer Segmentation (KMeans Clustering) ---
# Determine optimal number of clusters (e.g., using the Elbow method - this is a simplification)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png')
print("Plot saved to elbow_method.png")
# Based on the elbow method (visual inspection), let's choose 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)
# --- 4. Churn Analysis by Segment ---
churn_by_cluster = df.groupby('Cluster')['Churn'].mean()
print("\nChurn Rate by Cluster:")
print(churn_by_cluster)
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.barplot(x=churn_by_cluster.index, y=churn_by_cluster.values)
plt.title('Churn Rate by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Churn Rate')
plt.savefig('churn_by_segment.png')
print("Plot saved to churn_by_segment.png")
# --- 6.  Further Analysis (Example):  ---
#  You could further analyze the characteristics of each cluster to understand why they churn.
#  For example, you could calculate the average Recency, Frequency, and MonetaryValue for each cluster.
#Example:
cluster_characteristics = df.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean()
print("\nAverage Customer Characteristics by Cluster:")
print(cluster_characteristics)