import numpy as np
import pandas as pd
import pickle
import random

# ========================
# CONFIG
# ========================

FEATURES_PATH = "modelV2/data/processed/features_scaled.npy"
METADATA_PATH = "modelV2/data/processed/metadata.csv"
MODEL_PATH = "modelV2/data/processed/kmeans_model.pkl"

N_RECOMMENDATIONS = 10

# Emoción → cluster (ajustado a tus perfiles)
EMOTION_TO_CLUSTER = {
    "happy": 0,
    "joy": 0,
    "sad": 1,
    "melancholy": 1,
    "angry": 2,
    "energetic": 2,
    "intense": 2
}

# ========================
# LOAD DATA
# ========================

print("Loading model and data...")

X = np.load(FEATURES_PATH)
metadata = pd.read_csv(METADATA_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

labels = model.predict(X)
metadata["cluster"] = labels

print("Data loaded successfully")

# ========================
# RECOMMENDER FUNCTION
# ========================

def recommend_by_emotion(emotion, n=N_RECOMMENDATIONS):
    emotion = emotion.lower()

    if emotion not in EMOTION_TO_CLUSTER:
        raise ValueError(f"Emotion '{emotion}' not supported")

    cluster_id = EMOTION_TO_CLUSTER[emotion]

    cluster_songs = metadata[metadata["cluster"] == cluster_id]

    if len(cluster_songs) == 0:
        raise RuntimeError("No songs found for this cluster")

    recommendations = cluster_songs.sample(
        n=min(n, len(cluster_songs)),
        random_state=random.randint(0, 9999)
    )

    return recommendations[["name", "artists", "year"]]


# ========================
# MANUAL TEST
# ========================

if __name__ == "__main__":
    emotion = "happy"
    print(f"\nRecommendations for emotion: {emotion}\n")
    recs = recommend_by_emotion(emotion)
    print(recs.to_string(index=False))
