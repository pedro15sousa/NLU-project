import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import joblib


# Load preprocessed data
base_path = Path(__file__).parent
repo_path = (base_path / "../data/processed").resolve()
train_df = pd.read_csv(repo_path / 'train.csv')
val_df = pd.read_csv(repo_path / 'val.csv')

# Transform the text data to a sparse matrix of word counts using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_df['Tweets'])
X_val = vectorizer.transform(val_df['Tweets'])

# Save vectorizer
vectorizer_path = (base_path / "../utils").resolve()
joblib.dump(vectorizer, vectorizer_path / 'vectorizer.joblib')

# Convert the author labels to numerical values
authors = train_df['AuthorID'].unique()
author2id = {author: i for i, author in enumerate(authors)}
y_train = np.array([author2id[author] for author in train_df['AuthorID']])
y_val = np.array([author2id[author] for author in val_df['AuthorID']])


# Train the SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Save the model to disk
model_path = (base_path / "../models").resolve()
joblib.dump(clf, model_path / 'svm_model.joblib')
