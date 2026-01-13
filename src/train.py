import os
import pickle
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model import crear_modelo


def main():
    # Load processed dataset
    data_path = "data/processed/spotify_processed.csv"
    df = pd.read_csv(data_path)

    # Features and target
    X = df[['valence', 'energy', 'danceability', 'acousticness', 'year']]
    y = df['emotions']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train model
    model = crear_modelo()
    model.fit(X_train_scaled, y_train)

    # Pack everything in one object
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder
    }

    # Save model
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/emotion_music_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    
    time.sleep(500)
    

    print("✅ Model trained and saved successfully.")


if __name__ == "__main__":
    main()
