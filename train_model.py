import pandas as pd
import pickle
import os

from sklearn.cluster import AgglomerativeClustering

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train model
model = AgglomerativeClustering(
    n_clusters=5,
    linkage='ward'
)

model.fit(X)

# Save model
os.makedirs("models", exist_ok=True)

with open("models/hierarchical_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Hierarchical Clustering Model Saved!")