from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, CLICK_LABEL, CONVERSION_LABEL, NUMERIC_FEATURES, TEXT_FEATURE
from .evaluation import evaluate_predictions
from .ranking import business_score


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("text", TfidfVectorizer(max_features=220, ngram_range=(1, 2)), TEXT_FEATURE),
        ]
    )


def _build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=400,
                    solver="liblinear",
                    class_weight="balanced",
                ),
            ),
        ]
    )


@dataclass
class BaselineArtifacts:
    click_model: Pipeline
    conversion_model: Pipeline

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "BaselineArtifacts":
        with path.open("rb") as handle:
            return pickle.load(handle)


def train_baselines(train_frame: pd.DataFrame) -> BaselineArtifacts:
    click_model = _build_classifier()
    click_model.fit(train_frame, train_frame[CLICK_LABEL].to_numpy())

    clicked = train_frame[train_frame[CLICK_LABEL] == 1]
    conversion_source = clicked if (not clicked.empty and clicked[CONVERSION_LABEL].nunique() >= 2) else train_frame

    conversion_model = _build_classifier()
    conversion_model.fit(conversion_source, conversion_source[CONVERSION_LABEL].to_numpy())
    return BaselineArtifacts(click_model=click_model, conversion_model=conversion_model)


def predict_baselines(artifacts: BaselineArtifacts, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctr_scores = artifacts.click_model.predict_proba(frame)[:, 1]
    conversion_condition_scores = artifacts.conversion_model.predict_proba(frame)[:, 1]
    cvr_scores = ctr_scores * conversion_condition_scores
    scores = business_score(frame, ctr_scores, cvr_scores)
    return ctr_scores, cvr_scores, scores


def evaluate_baselines(artifacts: BaselineArtifacts, frame: pd.DataFrame) -> dict[str, float]:
    ctr_scores, cvr_scores, scores = predict_baselines(artifacts, frame)
    return evaluate_predictions(frame, ctr_scores, cvr_scores, scores)
