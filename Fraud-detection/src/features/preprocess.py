import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features based on notebook logic."""
    try:
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')

        df['sender_balance_difference'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['receiver_balance_difference'] = df['newbalanceDest'] - df['oldbalanceDest']

        df['amount_over_sender'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['amount_over_receiver'] = df['amount'] / (df['oldbalanceDest'] + 1)

        df['is_large_transaction'] = (df['amount'] > 200_000).astype(int)

        df = df.drop(columns=[
            'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ])

        return df
    except Exception as e:
        logger.exception("Feature engineering failed")
        raise e


def build_transformer(X: pd.DataFrame):
    """Create ColumnTransformer."""
    numeric = X.select_dtypes(include=['int64', 'float64', 'int16', 'float32']).columns
    categorical = X.select_dtypes(include=['object']).columns

    transformer = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('scaler', StandardScaler(), numeric)
    ])

    return transformer


def save_transformer(transformer, path: str):
    joblib.dump(transformer, path)