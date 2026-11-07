import re
import unicodedata

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# Initialize tools once
tokenizer = TweetTokenizer(
    preserve_case=False,
    strip_handles=False
)

stemmer = SnowballStemmer("english")


def clean_text(text):

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", str(text))

    # 2. Case folding
    text = text.casefold()

    return text
