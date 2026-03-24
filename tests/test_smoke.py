from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_ranker.config import DataConfig, TrainingConfig
from smart_ranker.pipeline import run_pipeline


def test_pipeline_smoke(tmp_path: Path) -> None:
    metrics = run_pipeline(
        output_dir=str(tmp_path / "artifacts"),
        data_config=DataConfig(
            requests=120,
            candidates_per_request=8,
            user_count=120,
            item_count=80,
            seed=7,
        ),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=64,
            patience=2,
        ),
    )
    assert "baseline" in metrics
    assert "deep_model" in metrics
    assert metrics["deep_model"]["ndcg_at_10"] >= 0.0
