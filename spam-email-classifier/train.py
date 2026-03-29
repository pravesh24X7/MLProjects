# This file contains the model training pipeline script for Spam classifier

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# loading data
data = pd.read_csv("./data/dataset.csv")

# preprocessing step
data["Category"] = data["Category"].map(
    {
        "ham" : 0,
        "spam" : 1,
    }
)

X = data["Message"]
y = data["Category"]

# split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state=39,
                                                   test_size=0.3)

# training pipeline
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("logistic_regression_clf", LogisticRegression(class_weight="balanced"))
])

# begin training
model_pipeline.fit(X_train, y_train)

# before saving model, check if save directoy exists or not.
os.makedirs("model", exist_ok=True)

# save model
joblib.dump(model_pipeline, "model/spam_email_model.pkl")
print("[*] Model training completed")
