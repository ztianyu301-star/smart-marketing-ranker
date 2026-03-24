from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import CATEGORICAL_FEATURES, TrainingConfig
from .evaluation import evaluate_predictions
from .features import FeatureProcessor
from .ranking import business_score


class RankingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, processor: FeatureProcessor):
        self.frame = frame.reset_index(drop=True)
        transformed = processor.transform(self.frame)
        self.numeric = torch.tensor(transformed["numeric"], dtype=torch.float32)
        self.click = torch.tensor(transformed["click"], dtype=torch.float32)
        self.conversion = torch.tensor(transformed["conversion"], dtype=torch.float32)
        self.categorical = {
            key: torch.tensor(transformed[key], dtype=torch.long) for key in CATEGORICAL_FEATURES
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "numeric": self.numeric[index],
            "click": self.click[index],
            "conversion": self.conversion[index],
            **{key: value[index] for key, value in self.categorical.items()},
        }


class WideAndDeepRanker(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinality: dict[str, int],
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        embed_output_dim = 0
        for name, cardinality in categorical_cardinality.items():
            embedding_dim = min(16, max(4, cardinality // 2 + 1))
            self.embeddings[name] = nn.Embedding(cardinality + 1, embedding_dim)
            embed_output_dim += embedding_dim

        self.wide = nn.Linear(numeric_dim, 16)
        self.deep = nn.Sequential(
            nn.Linear(numeric_dim + embed_output_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        final_dim = 16 + hidden_dims[1]
        self.ctr_head = nn.Linear(final_dim, 1)
        self.cvr_head = nn.Linear(final_dim, 1)

    def forward(self, numeric: torch.Tensor, categorical: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_vectors = [self.embeddings[name](categorical[name]) for name in self.embeddings]
        embedded = torch.cat(embedding_vectors, dim=1)
        deep_input = torch.cat([numeric, embedded], dim=1)
        deep_features = self.deep(deep_input)
        wide_features = self.wide(numeric)
        combined = torch.cat([wide_features, deep_features], dim=1)
        return self.ctr_head(combined).squeeze(1), self.cvr_head(combined).squeeze(1)


@dataclass
class DeepModelArtifacts:
    processor: FeatureProcessor
    model_state: dict[str, torch.Tensor]
    training_config: dict[str, object]
    categorical_cardinality: dict[str, int]

    def build_model(self) -> WideAndDeepRanker:
        model = WideAndDeepRanker(
            numeric_dim=len(self.processor.numeric_columns),
            categorical_cardinality=self.categorical_cardinality,
            hidden_dims=tuple(self.training_config["hidden_dims"]),
            dropout=float(self.training_config["dropout"]),
        )
        model.load_state_dict(self.model_state)
        model.eval()
        return model

    def save(self, path: str) -> None:
        torch.save(
            {
                "processor": self.processor.state_dict(),
                "model_state": self.model_state,
                "training_config": self.training_config,
                "categorical_cardinality": self.categorical_cardinality,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DeepModelArtifacts":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            processor=FeatureProcessor.from_state_dict(payload["processor"]),
            model_state=payload["model_state"],
            training_config=dict(payload["training_config"]),
            categorical_cardinality=dict(payload["categorical_cardinality"]),
        )


def _build_loader(frame: pd.DataFrame, processor: FeatureProcessor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = RankingDataset(frame, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict_deep_model(
    artifacts: DeepModelArtifacts,
    frame: pd.DataFrame,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = artifacts.build_model()
    device = torch.device("cpu")
    model.to(device)
    loader = _build_loader(frame, artifacts.processor, batch_size=batch_size, shuffle=False)

    ctr_outputs: list[np.ndarray] = []
    cvr_outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            categorical = {name: batch[name] for name in CATEGORICAL_FEATURES}
            ctr_logits, cvr_logits = model(batch["numeric"], categorical)
            ctr_prob = torch.sigmoid(ctr_logits).cpu().numpy()
            cvr_given_click = torch.sigmoid(cvr_logits).cpu().numpy()
            ctr_outputs.append(ctr_prob)
            cvr_outputs.append(ctr_prob * cvr_given_click)

    ctr_scores = np.concatenate(ctr_outputs)
    cvr_scores = np.concatenate(cvr_outputs)
    scores = business_score(frame, ctr_scores, cvr_scores)
    return ctr_scores, cvr_scores, scores


def evaluate_deep_model(artifacts: DeepModelArtifacts, frame: pd.DataFrame) -> dict[str, float]:
    ctr_scores, cvr_scores, scores = predict_deep_model(artifacts, frame)
    return evaluate_predictions(frame, ctr_scores, cvr_scores, scores)


def train_deep_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[DeepModelArtifacts, dict[str, float]]:
    processor = FeatureProcessor.fit(train_frame)
    categorical_cardinality = {
        feature: max(processor.category_maps[feature].values(), default=0) for feature in CATEGORICAL_FEATURES
    }
    model = WideAndDeepRanker(
        numeric_dim=len(processor.numeric_columns),
        categorical_cardinality=categorical_cardinality,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    device = torch.device(config.device)
    model.to(device)

    train_loader = _build_loader(train_frame, processor, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ctr_loss_fn = nn.BCEWithLogitsLoss()
    cvr_loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_metric = float("-inf")
    epochs_without_improve = 0
    training_curve: dict[str, float] = {}

    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()
            categorical = {name: batch[name] for name in CATEGORICAL_FEATURES}
            ctr_logits, cvr_logits = model(batch["numeric"], categorical)
            ctr_loss = ctr_loss_fn(ctr_logits, batch["click"])

            clicked_mask = batch["click"] > 0
            if torch.any(clicked_mask):
                cvr_loss = cvr_loss_fn(cvr_logits[clicked_mask], batch["conversion"][clicked_mask])
            else:
                cvr_loss = torch.tensor(0.0, device=device)

            loss = 0.7 * ctr_loss + 0.3 * cvr_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        current_artifacts = DeepModelArtifacts(
            processor=processor,
            model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
            training_config={
                "hidden_dims": list(config.hidden_dims),
                "dropout": config.dropout,
            },
            categorical_cardinality=categorical_cardinality,
        )
        metrics = evaluate_deep_model(current_artifacts, validation_frame)
        composite = metrics["ctr_auc"] + metrics["cvr_auc"] + metrics["ndcg_at_10"]
        training_curve[f"epoch_{epoch + 1}_loss"] = float(np.mean(epoch_losses))
        training_curve[f"epoch_{epoch + 1}_validation_ndcg"] = metrics["ndcg_at_10"]

        if composite > best_metric:
            best_metric = composite
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.patience:
                break

    assert best_state is not None, "Training did not produce a valid model state."
    artifacts = DeepModelArtifacts(
        processor=processor,
        model_state=best_state,
        training_config={
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
        },
        categorical_cardinality=categorical_cardinality,
    )
    final_metrics = evaluate_deep_model(artifacts, validation_frame)
    final_metrics.update(training_curve)
    return artifacts, final_metrics
