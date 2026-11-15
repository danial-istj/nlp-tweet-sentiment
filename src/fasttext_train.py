 
import os
import pandas as pd
from gensim.models import FastText


def train_fasttext():
    print("Loading cleaned training data for fastText...")

    train_df = pd.read_csv(
        "data/processed/train_cleaned.csv"
    ).fillna("")

    print("Training data loaded successfully!")