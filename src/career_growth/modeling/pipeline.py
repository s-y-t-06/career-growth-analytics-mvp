"""Scikit-learn pipelines for churn prediction."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from career_growth import config
from career_growth.features.model_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_logistic_regression_pipeline(random_state: int = config.RANDOM_SEED) -> Pipeline:
    """Return a balanced Logistic Regression pipeline with one-hot encoding.

    Categorical features are one-hot encoded with infrequent category handling.
    Numeric features are imputed with the median and standardized. The classifier
    uses balanced class weights.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        min_frequency=0.01,
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline


def build_hist_gradient_boosting_pipeline(
    random_state: int = config.RANDOM_SEED,
) -> Pipeline:
    """Return a HistGradientBoostingClassifier pipeline with one-hot encoding.

    HistGradientBoostingClassifier can handle numeric features natively, but the
    categorical features are one-hot encoded for consistency and interpretability.
    Numeric features are imputed with the median.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        min_frequency=0.01,
    )
    numeric_transformer = SimpleImputer(strategy="median")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    max_iter=500,
                ),
            ),
        ]
    )
    return pipeline
