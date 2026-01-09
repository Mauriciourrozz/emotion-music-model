from sklearn.ensemble import RandomForestClassifier


def crear_modelo():
    """
    Create and return the emotion-music classification model.

    This is a preliminary model used to validate the ML pipeline.
    The emotion labels are currently random and will be replaced
    by a proper labeling strategy in the future.
    """
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    return modelo
