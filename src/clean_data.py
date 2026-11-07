from pathlib import Path
import pandas as pd

from src.cleaning import clean_text

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

splits = ["train", "validation", "test"]

for split in splits:

    input_file = RAW_DIR / f"{split}.csv"

    df = pd.read_csv(input_file)

    print(f"Cleaning {split}...")

    df["cleaned_text"] = df["text"].apply(clean_text)

    output_file = (
        PROCESSED_DIR / f"{split}_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved: {output_file}")
    print(f"Documents: {len(df)}")

print("All splits cleaned successfully!")