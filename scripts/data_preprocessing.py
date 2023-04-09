import re 
import string 
import nltk
from pathlib import Path
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split

def preprocess_tweet(tweet):
    # Define regex patterns for preprocessing
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    user_pattern = r'@[^\s]+'
    hashtag_pattern = r'#([^\s]+)'
    emoji_pattern = r'[^\x00-\x7F]+'

    # Remove URLs
    tweet = re.sub(url_pattern, '', tweet)
    # Remove usernames
    tweet = re.sub(user_pattern, '', tweet)
    # Remove hashtags
    tweet = re.sub(hashtag_pattern, '', tweet)
    # Remove emojis
    tweet = re.sub(emoji_pattern, '', tweet)
    # Tokenize tweet
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet)
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if
                       token.lower() not in stop_words and token.lower() not in string.punctuation]
    # Join tokens back into string
    return ' '.join(filtered_tokens)

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

#Preprocess tweets
df_raw['Tweets'] = df_raw['Tweets'].apply(preprocess_tweet)

# check for empty strings in the Tweets column after cleaning  (e.g. tweets that only had emojis or links)
empty_tweets_count = len(df_raw[df_raw['Tweets'] == ''])
print("Number of empty Tweets: ", empty_tweets_count)

# replace empty strings with NaN
df_raw = df_raw.replace('', np.nan)

# drop rows containing NaN values
df_raw = df_raw.dropna()

# Count the number of tweets per author
tweet_counts = df_raw['AuthorID'].value_counts()

# Check if there is a significant imbalance in the number of tweets per author
if tweet_counts.std() / tweet_counts.mean() > 0.5:
    # Resample the data to balance the number of tweets per author
    df_balanced = df_raw.groupby('AuthorID').apply(lambda x: x.sample(n=min(len(x), 1000)))
    print('Data was imbalanced and has been resampled.')
else:
    df_balanced = df_raw.copy()


# split the data into training, validation, and testing sets
train, test = train_test_split(df_balanced, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

# save the data to appropriate data folders
repo_path = (base_path / "../data/processed").resolve()
train.to_csv(repo_path / 'train.csv', index=False, encoding='utf-8')
val.to_csv(repo_path / 'val.csv', index=False)
test.to_csv(repo_path / 'test.csv', index=False)