import pandas as pd
import random

# ---------------------------
# Configuración
# ---------------------------

EMOTION_TO_CLUSTER = {
    "happy": 0,
    "upbeat": 0,
    "party": 0,

    "sad": 1,
    "calm": 1,
    "emotional": 1,

    "energetic": 2,
    "angry": 2,
    "intense": 2
}

DATA_PATH = "data/processed/clustered_songs.csv"


# ---------------------------
# Cargar datos
# ---------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# ---------------------------
# Recomendador principal
# ---------------------------

def recommend_songs(emotion: str, n: int = 10):
    emotion = emotion.lower()

    if emotion not in EMOTION_TO_CLUSTER:
        raise ValueError(f"Emotion '{emotion}' not supported")

    cluster = EMOTION_TO_CLUSTER[emotion]
    df = load_data()

    cluster_songs = df[df["cluster"] == cluster]

    if len(cluster_songs) == 0:
        raise ValueError("No songs found for this cluster")

    recommendations = cluster_songs.sample(
        n=min(n, len(cluster_songs)),
        random_state=random.randint(0, 9999)
    )

    return recommendations[["name", "artists"]]


# ---------------------------
# Test manual
# ---------------------------

if __name__ == "__main__":
    print("\n🎵 Recommendations for HAPPY:\n")
    print(recommend_songs("happy", 5))
