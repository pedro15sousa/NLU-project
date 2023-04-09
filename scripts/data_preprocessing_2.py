import re 
import string 
import nltk
from pathlib import Path
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split


# Load raw tweets dataset
base_path = Path(__file__).parent
file_path = (base_path / "../data/raw/raw_tweets.csv").resolve()
df_raw = pd.read_csv(file_path)

# Separate data by author ID
authors = df_raw.AuthorID.unique()
author_data = []
for author in authors:
    author_tweets = df_raw[df_raw['AuthorID'] == author]
    author_data.append(author_tweets)

# Remove authors with fewer than 10 tweets
author_data = [author for author in author_data if author.shape[0] >= 10]

# Combine the author data into a single DataFrame
df_raw = pd.concat(author_data)

# Count the number of tweets per author
tweet_counts = df_raw['AuthorID'].value_counts()

# Check if there is a significant imbalance in the number of tweets per author
if tweet_counts.std() / tweet_counts.mean() > 0.5:
    # Resample the data to balance the number of tweets per author
    df_balanced = df_raw.groupby('AuthorID').apply(lambda x: x.sample(n=min(len(x), 1000)))
    print('Data was imbalanced and has been resampled.')
else:
    df_balanced = df_raw.copy()

possible_labels = df_raw['AuthorID'].unique()
#convert labels into numeric values
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

df_raw['label'] = df_raw.AuthorID.replace(label_dict)

#find the maximum length
max_len = max([len(sent) for sent in df_raw.Tweets])

#train test split
X_train, X_val, y_train, y_val = train_test_split(df_raw.index.values,
                                                   df_raw.label.values,
                                                   test_size = 0.15,
                                                   random_state = 17,
                                                   stratify = df_raw.label.values)


df_raw['data_type'] = ["not_set"] * df_raw.shape[0]
df_raw.loc[X_train, 'data_type'] = 'train'
df_raw.loc[X_val, 'data_type'] = 'val'
df_raw.drop(df_raw.columns[0], axis=1, inplace=True)

#groupby count
print(df_raw.groupby(['AuthorID', 'label', 'data_type']).count())


repo_path = (base_path / "../data/processed").resolve()
df_raw.to_csv(repo_path / 'total.csv', index=False)
