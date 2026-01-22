import numpy as np
import pandas as pd
import pickle

FEATURES = ["valence", "energy", "danceability", "tempo"]

print("Loading data...")

X = np.load("data/processed/features_scaled.npy")
metadata = pd.read_csv("data/processed/metadata.csv")

with open("data/processed/kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------
# ASSIGN CLUSTERS
# --------------------
labels = model.predict(X)
metadata["cluster"] = labels

# --------------------
# RECOVER ORIGINAL SCALE
# --------------------
# Volvemos a cargar el dataset original para interpretar valores reales
raw = pd.read_csv("data/raw/spotify_dataset_1921_2020.csv")

# Reaplicamos el MISMO filtro que preprocess
raw = raw[
    (raw["speechiness"] < 0.33) &
    (raw["instrumentalness"] < 0.5) &
    (raw["duration_ms"] > 60000) &
    (raw["duration_ms"] < 600000) &
    (raw["popularity"] > 20) &
    (raw["year"] >= 1990)
].reset_index(drop=True)

raw["cluster"] = labels

# --------------------
# CLUSTER PROFILES
# --------------------
print("\nCluster profiles (mean values):\n")

cluster_profiles = raw.groupby("cluster")[FEATURES].mean()
print(cluster_profiles)

# --------------------
# SAMPLE SONGS PER CLUSTER
# --------------------
print("\nSample songs per cluster:\n")

for cluster_id in sorted(raw["cluster"].unique()):
    print(f"\n--- Cluster {cluster_id} ---")
    sample = raw[raw["cluster"] == cluster_id].sample(5, random_state=42)
    print(sample[["name", "artists"]])
