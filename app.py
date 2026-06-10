import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Hierarchical Clustering")

# Load trained model
with open("models/hierarchical_model.pkl", "rb") as f:
    model = pickle.load(f)

income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0
)

score = st.number_input(
    "Spending Score",
    min_value=0.0,
    max_value=100.0
)

if st.button("Predict Cluster"):

    # Reload dataset
    df = pd.read_csv("data/Mall_Customers.csv")

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Add new point
    X_new = pd.concat(
        [
            X,
            pd.DataFrame(
                [[income, score]],
                columns=X.columns
            )
        ],
        ignore_index=True
    )

    from sklearn.cluster import AgglomerativeClustering

    temp_model = AgglomerativeClustering(
        n_clusters=5,
        linkage='ward'
    )

    labels = temp_model.fit_predict(X_new)

    cluster = labels[-1]

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )