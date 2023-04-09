import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import joblib

# Load the pre-processed datasets
base_path = Path(__file__).parent
repo_path = (base_path / "../data/processed").resolve()
test_df = pd.read_csv(repo_path / 'test.csv')

# Load saved vectorizer
vectorizer_path = (base_path / "../utils").resolve()
vectorizer = joblib.load(vectorizer_path / 'vectorizer.joblib')

# Transform the text data to a sparse matrix of word counts using CountVectorizer
X_test = vectorizer.transform(test_df['Tweets'])
# print(X_test.shape)

# Convert the author labels to numerical values
authors = test_df['AuthorID'].unique()
author2id = {author: i for i, author in enumerate(authors)}
y_test = np.array([author2id[author] for author in test_df['AuthorID']])

# Load the SVM model from disk
model_path = (base_path / "../models").resolve()
clf = joblib.load(model_path / 'svm_model.joblib')

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Convert the numerical predictions back to author labels
id2author = {i: author for author, i in author2id.items()}
y_pred_labels = np.array([id2author[i] for i in y_pred])
y_test_labels = np.array([id2author[i] for i in y_test])

# Print classification report
print(classification_report(y_test_labels, y_pred_labels))

