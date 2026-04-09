import joblib
import pandas as pd


class FraudPredictor:
    def __init__(self, model_path: str, transformer_path: str):
        self.model = joblib.load(model_path)
        self.transformer = joblib.load(transformer_path)

    def predict(self, df: pd.DataFrame):
        X = df.drop(columns=['isFraud'], errors='ignore')
        X_transformed = self.transformer.transform(X)
        return self.model.predict(X_transformed)
