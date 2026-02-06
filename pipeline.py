from modelV1.predict_emotion import predict_emotion
from modelV2.src.recommender import recommend_by_emotion

# ========================
# EMOTION MAPPING
# ========================

EKMAN_TO_RECOMMENDER = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "intense"
}

# ========================
# PIPELINE
# ========================

def recommend_song_from_text(text):
    """
    Takes a text input, detects the emotion, and recommends ONE song.
    """
    detected_emotion = predict_emotion(text)

    if detected_emotion not in EKMAN_TO_RECOMMENDER:
        raise ValueError(f"Emotion '{detected_emotion}' not supported")

    mapped_emotion = EKMAN_TO_RECOMMENDER[detected_emotion]

    recommendation = recommend_by_emotion(mapped_emotion, n=1)

    return detected_emotion, recommendation.iloc[0]


# ========================
# CLI
# ========================

if __name__ == "__main__":
    print("\n🎧 EMOMU - Emotional Music Recommender")
    print("Type a sentence and press Enter (type 'exit' to quit)\n")

    while True:
        text = input("How do you feel today? ")
        if text.lower() in ["exit", "quit"]:
            break

        emotion, song = recommend_song_from_text(text)

        print(f"\nDetected emotion: {emotion}")
        print("Recommended song:")
        print(f"🎵 {song['name']} — {song['artists']}\n")
