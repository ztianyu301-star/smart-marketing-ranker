from dataclasses import dataclass, field


NUMERIC_FEATURES = [
    "hour",
    "is_weekend",
    "user_age",
    "user_activity",
    "historical_ctr",
    "historical_cvr",
    "price",
    "bid",
    "merchant_quality",
    "freshness",
    "discount_rate",
    "query_length",
    "title_length",
    "token_overlap",
    "category_match",
    "price_gap",
    "semantic_score",
    "position_index",
]

CATEGORICAL_FEATURES = [
    "device",
    "city_tier",
    "gender",
    "query_category",
    "item_category",
]

TEXT_FEATURE = "combined_text"
CLICK_LABEL = "click_label"
CONVERSION_LABEL = "conversion_label"


@dataclass
class DataConfig:
    requests: int = 1500
    candidates_per_request: int = 12
    user_count: int = 700
    item_count: int = 500
    seed: int = 42


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.15
    patience: int = 3
    device: str = "cpu"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
