import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/raw/spotify_dataset_1921_2020.csv"

FEATURES = ["valence", "energy", "danceability", "tempo"]
META = ["name", "artists", "year"]

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# ------------------------
# FILTRO FUERTE (MÚSICA REAL)
# ------------------------
df = df[
    (df["speechiness"] < 0.33) &
    (df["instrumentalness"] < 0.5) &
    (df["duration_ms"] > 60000) &     # > 1 min
    (df["duration_ms"] < 600000) &    # < 10 min
    (df["popularity"] > 20) &
    (df["year"] >= 1990)
].reset_index(drop=True)

print("Filtered rows:", len(df))

X = df[FEATURES]
metadata = df[META]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.save("data/processed/features_scaled.npy", X_scaled)
metadata.to_csv("data/processed/metadata.csv", index=False)

print("Features shape:", X_scaled.shape)
print("Metadata shape:", metadata.shape)
