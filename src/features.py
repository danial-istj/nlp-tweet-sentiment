import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(train_path="data/processed/train_cleaned.csv", 
                           val_path="data/processed/validation_cleaned.csv", 
                           test_path="data/processed/test_cleaned.csv"):
    print("Loading cleaned datasets...")
    train_df = pd.read_csv(train_path).fillna("")
    val_df = pd.read_csv(val_path).fillna("")
    test_df = pd.read_csv(test_path).fillna("")

    # Initialize TF-IDF Vectorizer with unigrams, bigrams, and trigrams
    print("Initializing TF-IDF Vectorizer (ngram_range=(1, 3), max_features=10000)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)

    # Fit and transform training text
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_val = vectorizer.transform(val_df["cleaned_text"])
    X_test = vectorizer.transform(test_df["cleaned_text"])

   