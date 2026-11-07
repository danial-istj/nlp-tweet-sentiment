3.11.9
import pandas as pd

train = pd.read_csv("data/raw/train.csv")

print(train.head())
print(train.shape)
print(train["label_name"].value_counts())