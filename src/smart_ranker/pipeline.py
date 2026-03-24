from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .baselines import evaluate_baselines, train_baselines
from .config import DataConfig, PipelineConfig, TrainingConfig
from .data import generate_synthetic_ranking_data
from .model import evaluate_deep_model, predict_deep_model, train_deep_model
from .ranking import build_ranked_candidates


def run_pipeline(
    output_dir: str,
    data_config: DataConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict[str, dict[str, float]]:
    data_config = data_config or DataConfig()
    training_config = training_config or TrainingConfig()
    config = PipelineConfig(data=data_config, training=training_config)
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_ranking_data(data_config)
    train_frame = dataset[dataset["split"] == "train"].reset_index(drop=True)
    validation_frame = dataset[dataset["split"] == "validation"].reset_index(drop=True)
    test_frame = dataset[dataset["split"] == "test"].reset_index(drop=True)

    baseline_artifacts = train_baselines(train_frame)
    deep_artifacts, validation_training_metrics = train_deep_model(
        train_frame,
        validation_frame,
        training_config,
    )

    baseline_metrics = evaluate_baselines(baseline_artifacts, test_frame)
    deep_metrics = evaluate_deep_model(deep_artifacts, test_frame)
    deep_metrics.update(
        {f"validation_{key}": value for key, value in validation_training_metrics.items()}
    )

    baseline_artifacts.save(artifact_dir / "baseline.pkl")
    deep_artifacts.save(str(artifact_dir / "deep_model.pt"))
    dataset.head(400).to_csv(artifact_dir / "sample_dataset.csv", index=False)

    sample_request_id = test_frame["request_id"].iloc[0]
    sample_candidates = test_frame[test_frame["request_id"] == sample_request_id].copy()
    ctr_scores, cvr_scores, _ = predict_deep_model(deep_artifacts, sample_candidates)
    demo_ranking = build_ranked_candidates(sample_candidates, ctr_scores, cvr_scores)
    demo_ranking.to_json(artifact_dir / "demo_ranking.json", orient="records", indent=2)

    metrics = {
        "config": asdict(config),
        "baseline": baseline_metrics,
        "deep_model": deep_metrics,
    }
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    return {
        "baseline": baseline_metrics,
        "deep_model": deep_metrics,
    }
