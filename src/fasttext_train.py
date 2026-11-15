 
import os
import pandas as pd
from gensim.models import FastText


def train_fasttext():
    print("Loading cleaned training data for fastText...")

    train_df = pd.read_csv(
        "data/processed/train_cleaned.csv"
    ).fillna("")

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
    epochs=10,
    bucket=200000
)

    os.makedirs("models", exist_ok=True)

    ft_model.save("models/fasttext_tweet.model")

    print("fastText model saved successfully in models/")

    
    oov_word = "excitedd"

    print(
        f"Vector for OOV word '{oov_word}' generated via subwords:",
        ft_model.wv[oov_word][:5]
    )

    print(
        f"Most similar to '{oov_word}':",
        ft_model.wv.most_similar(oov_word, topn=3)
    )


if __name__ == "__main__":
    train_fasttext()
