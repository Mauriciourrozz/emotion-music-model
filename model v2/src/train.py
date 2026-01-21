import numpy as np
import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans

print("Loading features...")
X = np.load("data/processed/features_scaled.npy")

print("Loading metadata...")
metadata = pd.read_csv("data/processed/metadata.csv")

K = 3  # JUSTIFICADO POR SILHOUETTE

print(f"Training KMeans with k={K}...")
model = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=20
)

labels = model.fit_predict(X)

# Guardar modelo
with open("data/processed/kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Crear dataset final clusterizado
clustered_df = metadata.copy()
clustered_df["cluster"] = labels

# Guardar CSV que usa validation y recommender
clustered_df.to_csv(
    "data/processed/clustered_songs.csv",
    index=False
)

print("Training complete")
print("Clusters:", np.bincount(labels))
print("clustered_songs.csv creado correctamente")
