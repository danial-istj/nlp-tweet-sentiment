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

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save vectorizer and feature matrices
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump((X_train, train_df["label"]), "data/processed/X_train_tfidf.pkl")
    joblib.dump((X_val, val_df["label"]), "data/processed/X_val_tfidf.pkl")
    joblib.dump((X_test, test_df["label"]), "data/processed/X_test_tfidf.pkl")

    print(f"TF-IDF Extraction Complete! Train Shape: {X_train.shape}, Vocabulary Size: {len(vectorizer.vocabulary_)}")

if __name__ == "__main__":
    extract_tfidf_features()