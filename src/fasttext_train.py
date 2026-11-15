 
import os
import pandas as pd
from gensim.models import FastText


def train_fasttext():
    print("Loading cleaned training data for fastText...")

    train_df = pd.read_csv(
        "data/processed/train_cleaned.csv"
    ).fillna("")

    print("Training data loaded successfully!")

    sentences = [
        text.split()
        for text in train_df["cleaned_text"].tolist()
        if text.strip()
    ]

    print(f"Total sentences for fastText training: {len(sentences)}")

    print("Training fastText model (capturing subword structures)...")

    ft_model = FastText(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=10
    )