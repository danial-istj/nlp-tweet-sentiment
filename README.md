
# Twitter Sentiment Analysis

## 1. Project Overview

This project focuses on sentiment classification
of English-language social media text.

The objective is to classify tweets into:
- Negative
- Neutral
- Positive

## 2. Dataset

Dataset: TweetEval - Sentiment

Source:
https://huggingface.co/datasets/cardiffnlp/tweet_eval

Configuration: sentiment

Total documents: 59,899

Training: 45,615
Validation: 2,000
Testing: 12,284

Features:
- text
- label

Prediction target: label

Language: English

License:
The sentiment subset is identified as
CC BY 3.0 in the dataset card.
Applicable Twitter terms and source dataset
conditions must also be followed.

## 3. Project Structure

data/
notebooks/
src/
reports/

## 4. Environment Setup (Use Powershell terminal for Commands compatiblity)

Python 3.11.9

Create virtual environment:

python -m venv .venv

Activate:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  ((if persmissions restricted))

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

## 5. Download Dataset

python -m src.download_data

## 6. Run Cleaning Pipeline

# Download required NLTK data
python -c "import nltk; nltk.download('stopwords')"

python -m src.clean_data

## 7. Notebooks

01_exploration.ipynb
02_tokenization.ipynb

## 8. Output

The processed dataset is saved in:

data/processed/

## 9. Team Members

Add all four members and their responsibilities.

Muhammad Danial             L1F23BSCS1073
Syed Muhammad Ali Hussain   L1F23BSCS1063
Ali Haider                  L1F23BSCS1043
Ayesha Mushtaq              L1F22BSCS0182

## 10. References and Acknowledgements

TweetEval:
https://github.com/cardiffnlp/tweeteval

Dataset:
https://huggingface.co/datasets/cardiffnlp/tweet_eval

Reference paper:
TweetEval: Unified Benchmark and Comparative
Evaluation for Tweet Classification (2020).

AI assistance:
ChatGPT was used for guidance, code explanations,
debugging assistance and documentation support.
All code was reviewed, executed and validated
by the team.
