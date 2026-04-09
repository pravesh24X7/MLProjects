import os
from dotenv import load_dotenv
from src.data.data_loader import load_data, initial_cleaning
from src.features.preprocess import feature_engineering, build_transformer
from src.models.train import train_model

load_dotenv()

RAW_PATH = os.environ['DATA_PATH']
MODEL_PATH = os.environ['MODEL_PATH']
TRANSFORMER_PATH = os.environ['TRANSFORMER_PATH']


def main():
    df = load_data(RAW_PATH)
    df = initial_cleaning(df)
    df = feature_engineering(df)

    transformer = build_transformer(df.drop(columns=['isFraud']))

    train_model(df, transformer, MODEL_PATH, TRANSFORMER_PATH)


if __name__ == "__main__":
    main()