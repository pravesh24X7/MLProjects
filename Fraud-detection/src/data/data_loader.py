import pandas as pd
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: str, chunk_size: int = 100_000) -> pd.DataFrame:
    """Load large CSV using chunking."""
    try:
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception("Error loading data")
        raise e


def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns and optimize dtypes."""
    try:
        df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

        df['step'] = df['step'].astype('int16')
        float_cols = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ]
        for col in float_cols:
            df[col] = df[col].astype('float32')

        df['isFraud'] = df['isFraud'].astype('int8')

        return df
    except Exception as e:
        logger.exception("Error in cleaning")
        raise e