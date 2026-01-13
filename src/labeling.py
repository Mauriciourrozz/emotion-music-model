def assign_emotion(row):
    """
    Assign an emotion label to a song based on its musical features.

    Rules are heuristic and intended for a preliminary ML model.
    """

    valence = row["valence"]
    energy = row["energy"]
    danceability = row["danceability"]

    if valence >= 0.6 and energy >= 0.5:
        return "joy"

    elif valence <= 0.4 and energy <= 0.4:
        return "sadness"

    elif valence <= 0.4 and energy >= 0.6:
        return "anger"

    elif valence <= 0.5 and energy >= 0.5 and danceability <= 0.5:
        return "fear"

    else:
        return "neutral"
