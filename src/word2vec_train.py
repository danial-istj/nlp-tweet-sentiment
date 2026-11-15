import os
import pandas as pd
from gensim.models import Word2Vec

def train_word2vec():
    print("Loading cleaned training data for Word2Vec...")
    train_df = pd.read_csv("data/processed/train_cleaned.csv").fillna("")
    
    # Tokenize sentences (split by whitespace)
    sentences = [text.split() for text in train_df["cleaned_text"].tolist() if text.strip()]
    print(f"Total sentences for Word2Vec training: {len(sentences)}")

    # 1. Train CBOW Model (sg=0)
    print("Training Word2Vec CBOW model...")
    cbow_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, sg=0, epochs=10)
    
    # 2. Train Skip-gram Model (sg=1)
    print("Training Word2Vec Skip-gram model...")
    skipgram_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, sg=1, epochs=10)

    os.makedirs("models", exist_ok=True)
    cbow_model.save("models/word2vec_cbow.model")
    skipgram_model.save("models/word2vec_skipgram.model")
    print("Word2Vec models saved successfully in models/")

    # Test similarity if vocabulary permits
    test_word = "good"
    if test_word in cbow_model.wv:
        print(f"CBOW - Most similar to '{test_word}':", cbow_model.wv.most_similar(test_word, topn=3))
    if test_word in skipgram_model.wv:
        print(f"Skip-gram - Most similar to '{test_word}':", skipgram_model.wv.most_similar(test_word, topn=3))

if __name__ == "__main__":
    train_word2vec()