import pandas as pd
import json

"""
Loads the data and processes emotion labels,
mapping them to Ekman categories.
"""

def load_split(path):
    """Loads a data split from a TSV file."""
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["text", "labels", "id"]
    return df[["text", "labels"]]

"""Load data splits"""
train = load_split("data/train.tsv")
dev = load_split("data/dev.tsv")
test = load_split("data/test.tsv")

"""GO emotion ID mapping"""
id2emotion = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

"""Load GO emotion mapping to Ekman categories"""
with open("data/ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)

"""Create reverse mapping from GO to Ekman"""
go2ekman = {}
for ekman, emotions in ekman_mapping.items():
    for e in emotions:
        go2ekman[e] = ekman

"""Process data and map labels"""
def parse_label_ids(label_str):
    return [int(x) for x in str(label_str).split(",")]

"""Convert IDs to GO emotions"""
def ids_to_go_emotions(ids):
    return [id2emotion[i] for i in ids]

"""Map GO emotions to Ekman categories"""
def map_to_ekman(go_emotions):
    mapped = [go2ekman[e] for e in go_emotions if e in go2ekman]
    if len(mapped) == 0:
        return None
    return mapped[0]  # dominant emotion

"""Process a full split"""
def process_split(df):
    df = df.copy()

    df["label_ids"] = df["labels"].apply(parse_label_ids)
    df["go_emotions"] = df["label_ids"].apply(ids_to_go_emotions)
    df["emotion"] = df["go_emotions"].apply(map_to_ekman)
    df = df.dropna(subset=["emotion"])
    df = df[["text", "emotion"]]
    
    return df

# Process splits
train_processed = process_split(train)
dev_processed = process_split(dev)
test_processed = process_split(test)

# Save results
train_processed.to_csv("data/train_processed.csv", index=False)
dev_processed.to_csv("data/dev_processed.csv", index=False)
test_processed.to_csv("data/test_processed.csv", index=False)

print("Archivos generados:")
print("- train_processed.csv")
print("- dev_processed.csv")
print("- test_processed.csv")

print("\n=== COMPARACIÓN DE TAMAÑOS ===")
print(f"Train original: {len(train)}")
print(f"Train procesado: {len(train_processed)}")

print("\n=== DISTRIBUCIÓN FINAL (TRAIN) ===")
print(train_processed["emotion"].value_counts())
