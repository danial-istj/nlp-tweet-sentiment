
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

def analyze_bow():
    print("Loading cleaned training data for BoW analysis...")
    train_df = pd.read_csv("data/processed/train_cleaned.csv").fillna("")

    print("Fitting CountVectorizer (BoW)...")
    bow_vectorizer = CountVectorizer(max_features=10000)
    X_train_bow = bow_vectorizer.fit_transform(train_df["cleaned_text"])

    # Calculate matrix sparsity
    total_elements = X_train_bow.shape[0] * X_train_bow.shape[1]
    non_zero_elements = X_train_bow.nnz
    sparsity = (1.0 - (non_zero_elements / total_elements)) * 100

    print(f"--- BoW Analysis Report ---")
    print(f"Vocabulary Size: {len(bow_vectorizer.vocabulary_)}")
    print(f"Matrix Shape: {X_train_bow.shape}")
    print(f"Non-zero elements: {non_zero_elements}")
    print(f"Matrix Sparsity: {sparsity:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(bow_vectorizer, "models/bow_vectorizer.pkl")
    joblib.dump((X_train_bow, train_df["label"]), "data/processed/X_train_bow.pkl")
    print("BoW vectorizer and sparse matrix saved successfully.")

if __name__ == "__main__":
    analyze_bow()