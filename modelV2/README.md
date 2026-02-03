# Model V2 - Music Recommender

## Setup
```bash
pip install -r requirements.txt
```

## Run
**1. Preprocess data**
```bash
python src/preprocess.py
```

**2. Train clustering model**
```bash
python train.py
```

**3. Validate**
```bash
python src/validation.py
```

## Quick test
```bash
python -m src.recommender
```

## Hugging Face

If `modelV2/data/processed/*` is missing, `src/recommender.py` will
download `kmeans_model.pkl` and `metadata.csv` automatically from:
`1un4-13guis4m0/emotion-music-model`.

You can override the repo with:
```bash
export EMOMU_HF_REPO=your-user/your-repo
```
