import os
import pandas as pd
from gensim.models import Word2Vec

def train_word2vec():
    print("Loading cleaned training data for Word2Vec...")
    train_df = pd.read_csv("data/processed/train_cleaned.csv").fillna("")
    
    