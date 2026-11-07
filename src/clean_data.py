from pathlib import Path
import pandas as pd

from src.cleaning import clean_text

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)