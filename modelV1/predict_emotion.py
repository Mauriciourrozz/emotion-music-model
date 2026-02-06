from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""Predicts the emotion of a given text using a pretrained model."""

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_MODEL_NAME = "1un4-13guis4m0/emotion-distilbert-ekman"
DEFAULT_LOCAL_PATH = Path(__file__).resolve().parent / "emotion_model"

if DEFAULT_LOCAL_PATH.exists():
    model_source = str(DEFAULT_LOCAL_PATH)
else:
    model_source = HF_MODEL_NAME

model = AutoModelForSequenceClassification.from_pretrained(model_source)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_source)
id2label = model.config.id2label
MAX_LENGTH = 128

# Prediction function
def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return id2label[predicted_class_id]

# CLI (Command Line Interface)
if __name__ == "__main__":
    print("Emotion Detection (type 'exit' to quit)\n")

    while True:
        text = input("Texto: ")
        if text.lower() in ["exit", "quit"]:
            break

        emotion = predict_emotion(text)
        print(f"Emotion of the day: {emotion}\n")
