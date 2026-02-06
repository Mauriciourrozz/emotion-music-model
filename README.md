# Emotion-Based Music Recommendation System
---
This project combines **emotion detection from text** with a **music recommendation system**.

The system works in two stages:

1. A text model analyzes user input and predicts an emotion  
2. A music recommender suggests songs based on clusters linked to each emotion

---

## Project Structure
```bash
emotion-music-model/
│
├── modelV1/ # Emotion detection model (text → emotion)
│ ├── data/
│ ├── requirements.txt
│ ├── data_processed.py
│ ├── tokenize_data.py
│ ├── model_training.py
│ └── predict_emotion.py
│
├── modelV2/ # Music recommender (clustering)
│ ├── data/
│ │ ├── raw/
│ │ └── processed/
│ ├── src/
│ │ ├── preprocess.py
│ │ ├── validation.py
│ │ └── recommender.py
│ ├── train.py
│ └── requirements.txt
│
├── README.md
└── .gitignore
```

---

## How It Works

### Model 1 – Emotion Detection
- Takes a text input from the user
- Predicts an emotion (happy, sad, angry, etc.)

### Model 2 – Music Recommendation
- Uses Spotify audio features
- Applies KMeans clustering
- Each emotion is mapped to a cluster
- Random songs are sampled from the selected cluster

---

## Quickstart (End-to-End)
Run the full pipeline to go from text input → emotion → recommendation.
```bash
pip install -r modelV1/requirements.txt
pip install -r modelV2/requirements.txt
python pipeline.py
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Mauriciourrozz/emotion-music-model.git
cd emotion-music-model
```

### Install Model 1 dependencies
```bash
cd modelV1
pip install -r requirements.txt
```

### Install Model 2 dependencies
```bash
cd modelV2
pip install -r requirements.txt
```

---

## Running Model 1 (Emotion Detection)
**1. Process raw data**
```bash
cd modelV1
python data_processed.py
```

**2. Tokenize data**
```bash
python tokenize_data.py
```

**3. Train the model**
```bash
python model_training.py
```

**4. Predict emotion**
```bash
python predict_emotion.py
```

Notes:
- `predict_emotion.py` loads a local model from `modelV1/emotion_model` if it exists.
- If it doesn't exist, it falls back to the hosted model `1un4-13guis4m0/emotion-distilbert-ekman`.
- `modelV2/src/recommender.py` will use local `modelV2/data/processed/*` if present,
  otherwise it downloads artifacts from `1un4-13guis4m0/emotion-music-model` on Hugging Face.
  You can override the repo with `EMOMU_HF_REPO`.

## Running the Music Recommender (Model 2)
**1. Preprocess data**
```bash
cd modelV2
python src/preprocess.py
```

**2. Train clustering model**
```bash
python train.py
```

**3. Validate model**
```bash
python src/validation.py
```

**4. Run the app**
```bash
cd ..
python pipeline.py
```

---

## Example

Input text:
```bash
I am very happy because today is my birthday

You: i am very happy because today is my birthday

Detected emotion: joy
Recommended song:
🎵 Give a Little Whistle — ['Cliff Edwards', 'Dickie Jones'] (1992)
```

---

## Authors
Mauricio Urroz - https://github.com/Mauriciourrozz  
Luna Leguisamo - https://github.com/LunaLeguisamo

