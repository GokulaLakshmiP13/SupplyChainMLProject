import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# =====================================================
# SUPERVISED LEARNING – SHIPMENT DELAY PREDICTION
# =====================================================

print("\n========== SUPERVISED LEARNING ==========\n")

# Load dataset
data = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')
print("Original Dataset Shape:", data.shape)

# Select relevant columns
selected_columns = [
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Order Region',
    'Order Status',
    'Order Item Quantity',
    'Order Item Product Price',
    'Benefit per order',
    'Sales',
    'Order Item Discount Rate',
    'Market',
    'Category Name',
    'Late_delivery_risk'
]

data_supervised = data[selected_columns]
print("After Feature Selection:", data_supervised.shape)

# Encode categorical variables
data_supervised = pd.get_dummies(data_supervised, drop_first=True)
print("After Encoding:", data_supervised.shape)

# Split features & target
X = data_supervised.drop('Late_delivery_risk', axis=1)
y = data_supervised['Late_delivery_risk']

print("\nTarget Distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (only for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Logistic Regression ----------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("\n----- Logistic Regression -----")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n----- Random Forest -----")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
# ===============================
# FEATURE IMPORTANCE (Random Forest)
# ===============================

import matplotlib.pyplot as plt
import numpy as np

feature_importance = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:\n")
print(importance_df.head(10))

# Plot top 10 features
plt.figure(figsize=(8,5))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

# =====================================================
# UNSUPERVISED LEARNING – SHIPMENT PATTERN CLUSTERING
# =====================================================

print("\n========== UNSUPERVISED LEARNING ==========\n")

cluster_features = [
    'Days for shipment (scheduled)',
    'Order Item Quantity',
    'Order Item Product Price',
    'Benefit per order',
    'Sales',
    'Order Item Discount Rate'
]

cluster_data = data[cluster_features].copy()

# Scaling
scaler_cluster = StandardScaler()
cluster_scaled = scaler_cluster.fit_transform(cluster_data)

# ---------------- Elbow Method ----------------
wcss = []
for i in range(1, 11):
    kmeans_test = KMeans(n_clusters=i, random_state=42)
    kmeans_test.fit(cluster_scaled)
    wcss.append(kmeans_test.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# ---------------- K-Means Clustering ----------------
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(cluster_scaled)

cluster_data.loc[:, 'KMeans_Cluster'] = clusters

print("\nK-Means Cluster Distribution:\n")
print(cluster_data['KMeans_Cluster'].value_counts())

print("\nK-Means Cluster Means:\n")
print(cluster_data.groupby('KMeans_Cluster').mean())

# ---------------- Hierarchical Clustering ----------------
# Using only 5000 samples for dendrogram to avoid memory crash
linked = linkage(cluster_scaled[:5000], method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

hc = AgglomerativeClustering(n_clusters=4)
hc_labels = hc.fit_predict(cluster_scaled)

cluster_data.loc[:, 'Hierarchical_Cluster'] = hc_labels

print("\nHierarchical Cluster Distribution:\n")
print(cluster_data['Hierarchical_Cluster'].value_counts())

print("\nHierarchical Cluster Means:\n")
print(cluster_data.groupby('Hierarchical_Cluster').mean())

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========\n")