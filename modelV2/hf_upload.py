"""
Upload Model V2 artifacts to Hugging Face Hub.

Requires:
  pip install huggingface_hub
  huggingface-cli login
"""
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "1un4-13guis4m0/emotion-music-model"
REPO_TYPE = "model"
PRIVATE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

FILES_TO_UPLOAD = [
    DATA_DIR / "kmeans_model.pkl",
    DATA_DIR / "metadata.csv",
]

README_PATH = BASE_DIR / "hf_README.md"


def main():
    api = HfApi()
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=PRIVATE,
        exist_ok=True,
    )

    for file_path in FILES_TO_UPLOAD:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}. Generate it before uploading."
            )
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    if README_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(README_PATH),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )

    print("Upload complete.")


if __name__ == "__main__":
    main()
