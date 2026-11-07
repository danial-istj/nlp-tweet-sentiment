
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Dataset configuration
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"

# Output directory
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable labels
LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def download_dataset():
    print("Downloading TweetEval Sentiment dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG
    )

    for split_name, split_data in dataset.items():

        df = split_data.to_pandas()

        df["label_name"] = df["label"].map(LABELS)

        output_file = OUTPUT_DIR / f"{split_name}.csv"

        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

        print(
            f"{split_name}: {len(df)} documents saved"
        )

    print("Dataset download completed!")


if __name__ == "__main__":
    download_dataset()
