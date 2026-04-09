import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(df, transformer, model_path, transformer_path):
    try:
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']

        X_transformed = transformer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y.values,
            test_size=0.3,
            stratify=y
        )

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weight_dict = dict(zip(classes, weights))

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            n_jobs=-1,
            class_weight=class_weight_dict
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        logger.info("\n" + classification_report(y_test, preds))

        joblib.dump(model, model_path)
        joblib.dump(transformer, transformer_path)

        return model

    except Exception as e:
        logger.exception("Training failed")
        raise e
