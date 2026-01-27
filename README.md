# Emotion-Based Music Recommendation System
---
This project combines **emotion detection from text** with a **music recommendation system**

The system works in two stages:

1. A text model analyzes user input and predicts an emotion  
2. A music recommender suggests songs based on clusters linked to each emotion

---

## 📁 Project Structure

emotion-music-model/
│
├── modelV1/ # Emotion detection model (text → emotion)
│ ├── data/
│ ├── notebooks/
│ ├── scripts/
│ └── requirements.txt
│
├── modelV2/ # Music recommender (clustering)
│ ├── data/
│ │ ├── raw/
│ │ └── processed/
│ ├── preprocess.py
│ ├── train.py
│ ├── validation.py
│ ├── recommender.py
│ └── requirements.txt
│
├── README.md
└── .gitignore

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

## Installation

### Clone the repository:

```bash
git clone https://github.com/Mauriciourrozz/emotion-music-model.git
cd emotion-music-model
```

### Install Model 1 dependencies:
```bash
cd modelV1
pip install -r requirements.txt
```

### Install Model 2 dependencies:
```bash
cd modelV2
pip install -r requirements.txt
```

---

## Running the Music Recommender (Model 2)
**1. Preprocess data**
```bash
cd modelV2
python preprocess.py
```

**2. Train clustering model**
```bash
python train.py
```

**3. Validate model**
```bash
python validation.py
```

**4. running app**
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
