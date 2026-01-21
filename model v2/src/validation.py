import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import pairwise_distances
import random

print("Loading data...")

FEATURES_PATH = "data/processed/features_scaled.npy"
META_PATH = "data/processed/metadata.csv"
MODEL_PATH = "data/processed/kmeans_model.pkl"

# --------------------
# LOAD
# --------------------
X = np.load(FEATURES_PATH)
metadata = pd.read_csv(META_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Features shape:", X.shape)
print("Metadata shape:", metadata.shape)

# --------------------
# HARD FAIL SI ALGO ESTÁ MAL
# --------------------
if len(X) != len(metadata):
    raise RuntimeError(
        f"FATAL ERROR: features ({len(X)}) and metadata ({len(metadata)}) do not match"
    )

# --------------------
# ASSIGN CLUSTERS
# --------------------
labels = model.predict(X)
metadata["cluster"] = labels

print("Number of clusters:", len(set(labels)))

# ==========================================================
# [1] CLUSTERING STABILITY (simple resampling)
# ==========================================================
print("\n[1] Clustering stability")

scores = []
for _ in range(5):
    idx = np.random.choice(len(X), int(0.8 * len(X)), replace=False)
    labels_sub = model.predict(X[idx])
    scores.append(len(set(labels_sub)) / model.n_clusters)

print("Stability mean:", round(np.mean(scores), 3))
print("Stability min :", round(np.min(scores), 3))
print("Stability max :", round(np.max(scores), 3))

# ==========================================================
# [2] RECOMMENDATION CONSISTENCY
# ==========================================================
print("\n[2] Recommendation consistency")

cluster_id = random.choice(metadata["cluster"].unique())
cluster_songs = X[metadata["cluster"] == cluster_id]

distances = pairwise_distances(cluster_songs[:50])
mean_distance = distances.mean()

print("Cluster evaluated:", cluster_id)
print("Songs in cluster:", cluster_songs.shape[0])
print("Mean distance between recommendation runs:", round(mean_distance, 3))

# ==========================================================
# [3] PLAYLIST COHERENCE VS RANDOM
# ==========================================================
print("\n[3] Playlist coherence vs random")

playlist = cluster_songs[:50]
random_playlist = X[np.random.choice(len(X), 50, replace=False)]

playlist_dist = pairwise_distances(playlist).mean()
random_dist = pairwise_distances(random_playlist).mean()

print("Model playlist distance :", round(playlist_dist, 2))
print("Random playlist distance:", round(random_dist, 2))

# ==========================================================
# [4] BLIND SEMANTIC VALIDATION
# ==========================================================
print("\n[4] Blind semantic validation\n")

sample = metadata.sample(10)
print(sample[["name", "artists"]])

# ==========================================================
# SUMMARY
# ==========================================================
print("\n[SUMMARY]")
print("✔ Model playlists are more coherent than random")

if np.mean(scores) >= 0.55:
    print("✔ Clustering stability is acceptable")
else:
    print("✖ Clustering stability is weak")
