from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pickle
import random
from huggingface_hub import hf_hub_download


# config
BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features_scaled.npy"
METADATA_PATH = BASE_DIR / "data" / "processed" / "metadata.csv"
MODEL_PATH = BASE_DIR / "data" / "processed" / "kmeans_model.pkl"
HF_REPO_ID = os.getenv("EMOMU_HF_REPO", "1un4-13guis4m0/emotion-music-model")

N_RECOMMENDATIONS = 10

# cluster label
EMOTION_TO_CLUSTER = {
    "happy": 0,
    "joy": 0,
    "sad": 1,
    "melancholy": 1,
    "angry": 2,
    "energetic": 2,
    "intense": 2
}

def _get_file(path: Path, label: str, filename: str) -> Path:
    if path.exists():
        return path

    try:
        print(f"Downloading {label} from Hugging Face repo: {HF_REPO_ID}")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
        )
        return Path(downloaded)
    except Exception as exc:
        raise FileNotFoundError(
            f"Missing {label} at {path} and failed to download "
            f"from {HF_REPO_ID}. Upload artifacts or run preprocessing/training."
        ) from exc


@lru_cache(maxsize=1)
def _load_assets():
    features_path = _get_file(FEATURES_PATH, "features", "features_scaled.npy")
    metadata_path = _get_file(METADATA_PATH, "metadata", "metadata.csv")
    model_path = _get_file(MODEL_PATH, "kmeans model", "kmeans_model.pkl")

    X = np.load(features_path)
    metadata = pd.read_csv(metadata_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    labels = model.predict(X)
    metadata["cluster"] = labels

    return X, metadata, model

# recommend function

def recommend_by_emotion(emotion, n=N_RECOMMENDATIONS):
    _, metadata, _ = _load_assets()
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

# main

if __name__ == "__main__":
    emotion = "happy"
    print(f"\nRecommendations for emotion: {emotion}\n")
    recs = recommend_by_emotion(emotion)
    print(recs.to_string(index=False))
