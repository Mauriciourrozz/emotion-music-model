---
license: mit
tags:
- music
- recommendation
- clustering
- kmeans
- audio-features
---

# Emotion Music Model V2

This repository hosts the artifacts for the Model V2 recommender:

- `kmeans_model.pkl`: trained KMeans model
- `metadata.csv`: song metadata aligned with features

## How to use

Download files from this repo, place them in:

```
modelV2/data/processed/
```

Then run:

```bash
python -m modelV2.src.recommender
```

## Training

These artifacts were generated using the scripts in this project:

```bash
python modelV2/src/preprocess.py
python modelV2/train.py
```

## Notes

The recommender maps emotions to clusters and samples songs from the target cluster.
