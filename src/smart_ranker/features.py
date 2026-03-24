from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import CATEGORICAL_FEATURES, CLICK_LABEL, CONVERSION_LABEL, NUMERIC_FEATURES


@dataclass
class FeatureProcessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    category_maps: dict[str, dict[str, int]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> "FeatureProcessor":
        numeric_columns = numeric_columns or list(NUMERIC_FEATURES)
        categorical_columns = categorical_columns or list(CATEGORICAL_FEATURES)
        numeric_mean = {col: float(frame[col].mean()) for col in numeric_columns}
        numeric_std = {
            col: float(frame[col].std()) if float(frame[col].std()) > 1e-6 else 1.0
            for col in numeric_columns
        }
        category_maps = {}
        for column in categorical_columns:
            unique_values = sorted(frame[column].astype(str).unique().tolist())
            category_maps[column] = {value: index + 1 for index, value in enumerate(unique_values)}
        return cls(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            numeric_mean=numeric_mean,
            numeric_std=numeric_std,
            category_maps=category_maps,
        )

    def transform_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in self.numeric_columns:
            standardized = (frame[column].astype(float) - self.numeric_mean[column]) / self.numeric_std[column]
            columns.append(standardized.to_numpy(dtype=np.float32))
        return np.stack(columns, axis=1)

    def transform_categorical(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        encoded: dict[str, np.ndarray] = {}
        for column in self.categorical_columns:
            mapping = self.category_maps[column]
            encoded[column] = frame[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
        return encoded

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        payload = {
            "numeric": self.transform_numeric(frame),
            "click": frame[CLICK_LABEL].to_numpy(dtype=np.float32),
            "conversion": frame[CONVERSION_LABEL].to_numpy(dtype=np.float32),
        }
        payload.update(self.transform_categorical(frame))
        return payload

    def state_dict(self) -> dict[str, object]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "category_maps": self.category_maps,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "FeatureProcessor":
        return cls(
            numeric_columns=list(state["numeric_columns"]),
            categorical_columns=list(state["categorical_columns"]),
            numeric_mean=dict(state["numeric_mean"]),
            numeric_std=dict(state["numeric_std"]),
            category_maps={key: dict(value) for key, value in dict(state["category_maps"]).items()},
        )
