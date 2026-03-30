# This file contains the training pipeline of the Sentiment classification project done using NaiveBayes classifier.

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline

# loading dataset
data = pd.read_csv("./datasets/sentiment_analysis_dataset.csv")

# preprocessing
data.drop(columns=['Unnamed: 0', 'sentiment', 'id'], inplace=True)
data.dropna(inplace=True)

X = data['text']
y = data['label']

# split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values,
                                                    random_state=42,
                                                    test_size=0.3,
                                                    stratify=y)

# compute class weights
weights = compute_sample_weight(class_weight="balanced", y=y_train)

# training pipeline
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english",
                              lowercase=True,
                             min_df=5,
                             max_df=0.8,
                             max_features=10000,
                             ngram_range=(1,2))),
    ("mnb_clf", MultinomialNB( alpha=1.75 ))
])

# begin training model
model_pipeline.fit(X_train, y_train, mnb_clf__sample_weight=weights)

# before saving model, check if save directoy exists or not.
os.makedirs("saved_models", exist_ok=True)

# save model
joblib.dump(model_pipeline, "saved_models/sentiment_classifier.pkl")
print("[*] Model training completed")
