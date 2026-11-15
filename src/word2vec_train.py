import os
import pandas as pd
from gensim.models import Word2Vec

def train_word2vec():
    print("Loading cleaned training data for Word2Vec...")
    train_df = pd.read_csv("data/processed/train_cleaned.csv").fillna("")
    
    # Tokenize sentences (split by whitespace)
    sentences = [text.split() for text in train_df["cleaned_text"].tolist() if text.strip()]
    print(f"Total sentences for Word2Vec training: {len(sentences)}")

   