import pickle
import pandas as pd


FEATURES = ['valence', 'energy', 'danceability', 'acousticness', 'year']


def cargar_modelo():
    with open("../models/emotion_music_model.pkl", "rb") as f:
        return pickle.load(f)


def cargar_datos():
    return pd.read_csv("data/processed/spotify_processed.csv")


def recomendar_canciones(emocion, top_n=5):
    bundle = cargar_modelo()
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]

    df = cargar_datos()

    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    # Probabilities for each emotion
    proba = model.predict_proba(X_scaled)

    # Index of requested emotion
    emotion_index = list(label_encoder.classes_).index(emocion)

    # Score for the requested emotion
    df["score"] = proba[:, emotion_index]

    # Sort by score descending
    top_canciones = df.sort_values(by="score", ascending=False).head(top_n)

    resultados = []
    for _, row in top_canciones.iterrows():
        resultados.append({
            "artist": row["artists"],
            "year": row["year"],
            "emotion": emocion,
            "score": round(row["score"], 3)
        })

    return resultados



if __name__ == "__main__":
    emocion_usuario = "joy"
    recomendaciones = recomendar_canciones(emocion_usuario, top_n=5)

    print("🎵 Top recommendations:")
    for r in recomendaciones:
        print(r)
