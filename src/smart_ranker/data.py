from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig


CATEGORY_PROFILES = {
    "electronics": {
        "queries": [
            "gaming laptop discount",
            "student tablet value",
            "noise cancelling headset",
            "camera phone upgrade",
        ],
        "title_tokens": ["flagship", "fast", "warranty", "battery", "portable"],
        "price_range": (2800, 9800),
    },
    "beauty": {
        "queries": [
            "sensitive skin serum",
            "hydrating facial mask",
            "lipstick trending shade",
            "anti aging essence",
        ],
        "title_tokens": ["gentle", "hydrating", "premium", "repair", "giftbox"],
        "price_range": (80, 560),
    },
    "sports": {
        "queries": [
            "running shoes cushioning",
            "home fitness mat",
            "marathon energy gel",
            "basketball training set",
        ],
        "title_tokens": ["breathable", "durable", "training", "lightweight", "pro"],
        "price_range": (60, 1200),
    },
    "finance": {
        "queries": [
            "beginner wealth management",
            "family insurance plan",
            "credit card cashback",
            "retirement investment advice",
        ],
        "title_tokens": ["secure", "stable", "family", "flexible", "official"],
        "price_range": (1, 80),
    },
    "mother_baby": {
        "queries": [
            "baby diaper breathable",
            "infant milk powder stage 2",
            "stroller lightweight foldable",
            "baby toy sensory learning",
        ],
        "title_tokens": ["safe", "organic", "soft", "trusted", "growth"],
        "price_range": (40, 1600),
    },
    "travel": {
        "queries": [
            "weekend hotel package",
            "family trip luggage set",
            "travel skincare pack",
            "business travel backpack",
        ],
        "title_tokens": ["compact", "premium", "lightweight", "vacation", "bundle"],
        "price_range": (120, 2600),
    },
    "education": {
        "queries": [
            "ai course beginner",
            "math olympiad training",
            "english speaking bootcamp",
            "data analysis project class",
        ],
        "title_tokens": ["systematic", "live", "practice", "mentor", "advanced"],
        "price_range": (99, 4200),
    },
    "groceries": {
        "queries": [
            "healthy snack low sugar",
            "coffee beans arabica",
            "instant meal family pack",
            "organic fruit gift box",
        ],
        "title_tokens": ["fresh", "organic", "family", "premium", "daily"],
        "price_range": (15, 320),
    },
}

DEVICES = ["android", "ios", "pc"]
CITY_TIERS = ["tier1", "tier2", "tier3", "tier4"]
GENDERS = ["female", "male"]


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def _sample_category(rng: np.random.Generator) -> str:
    return rng.choice(list(CATEGORY_PROFILES.keys()))


def _token_overlap(query: str, title: str) -> float:
    query_tokens = set(query.split())
    title_tokens = set(title.split())
    if not query_tokens:
        return 0.0
    return len(query_tokens & title_tokens) / len(query_tokens)


def _build_users(config: DataConfig, rng: np.random.Generator) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = []
    for user_id in range(config.user_count):
        preferred_category = _sample_category(rng)
        users.append(
            {
                "user_id": f"user_{user_id:04d}",
                "user_age": int(rng.integers(21, 42)),
                "gender": rng.choice(GENDERS, p=[0.54, 0.46]),
                "city_tier": rng.choice(CITY_TIERS, p=[0.22, 0.31, 0.29, 0.18]),
                "device": rng.choice(DEVICES, p=[0.55, 0.28, 0.17]),
                "preferred_category": preferred_category,
                "user_activity": float(rng.uniform(0.2, 1.0)),
                "historical_ctr": float(rng.uniform(0.03, 0.22)),
                "historical_cvr": float(rng.uniform(0.01, 0.12)),
                "budget_level": float(rng.uniform(0.1, 1.0)),
            }
        )
    return users


def _build_items(config: DataConfig, rng: np.random.Generator) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    categories = list(CATEGORY_PROFILES.keys())
    for item_id in range(config.item_count):
        category = categories[item_id % len(categories)]
        profile = CATEGORY_PROFILES[category]
        query = rng.choice(profile["queries"])
        descriptors = rng.choice(profile["title_tokens"], size=2, replace=False)
        price_min, price_max = profile["price_range"]
        items.append(
            {
                "item_id": f"item_{item_id:04d}",
                "item_category": category,
                "title_text": f"{descriptors[0]} {query} {descriptors[1]}",
                "price": float(rng.uniform(price_min, price_max)),
                "merchant_quality": float(rng.uniform(0.45, 0.98)),
                "freshness": float(rng.uniform(0.1, 1.0)),
                "discount_rate": float(rng.uniform(0.0, 0.35)),
                "bid": float(rng.uniform(0.2, 3.0)),
            }
        )
    return items


def _request_split(request_index: int, total_requests: int) -> str:
    ratio = request_index / total_requests
    if ratio < 0.7:
        return "train"
    if ratio < 0.85:
        return "validation"
    return "test"


def _sample_query(category: str, rng: np.random.Generator) -> str:
    profile = CATEGORY_PROFILES[category]
    return rng.choice(profile["queries"])


def generate_synthetic_ranking_data(config: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    users = _build_users(config, rng)
    items = _build_items(config, rng)

    items_by_category: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        items_by_category.setdefault(item["item_category"], []).append(item)

    records: list[dict[str, Any]] = []
    all_categories = list(CATEGORY_PROFILES.keys())

    for request_index in range(config.requests):
        user = users[int(rng.integers(0, len(users)))]
        request_category = (
            user["preferred_category"] if rng.random() < 0.65 else rng.choice(all_categories)
        )
        query_text = _sample_query(request_category, rng)
        split = _request_split(request_index, config.requests)
        hour = int(rng.integers(0, 24))
        is_weekend = int(rng.random() < 0.3)

        positive_pool = items_by_category[request_category]
        negative_candidates: list[dict[str, Any]] = []
        while len(negative_candidates) < config.candidates_per_request * 2:
            negative_category = rng.choice([cat for cat in all_categories if cat != request_category])
            negative_pool = items_by_category[negative_category]
            negative_candidates.append(negative_pool[int(rng.integers(0, len(negative_pool)))])

        positive_count = max(2, int(config.candidates_per_request * 0.65))
        positives = list(rng.choice(positive_pool, size=positive_count, replace=False))
        negatives = list(
            rng.choice(negative_candidates, size=config.candidates_per_request - positive_count, replace=False)
        )
        candidate_items = positives + negatives
        rng.shuffle(candidate_items)

        for position_index, item in enumerate(candidate_items):
            category_match = float(item["item_category"] == request_category)
            overlap = _token_overlap(query_text, item["title_text"])
            price_gap = abs(user["budget_level"] - min(item["price"] / 10000.0, 1.0))
            semantic_score = (
                0.55 * category_match
                + 0.35 * overlap
                + 0.10 * item["merchant_quality"]
            )
            ctr_logit = (
                -1.6
                + 1.25 * category_match
                + 0.95 * overlap
                + 0.65 * user["historical_ctr"]
                + 0.45 * item["merchant_quality"]
                + 0.22 * item["discount_rate"]
                + 0.18 * item["freshness"]
                + 0.30 * semantic_score
                - 0.38 * price_gap
                - 0.12 * position_index
                + 0.08 * (hour >= 20)
                + 0.06 * is_weekend
            )
            ctr_probability = float(sigmoid(ctr_logit + rng.normal(0, 0.22)))
            click_label = int(rng.random() < ctr_probability)

            cvr_logit = (
                -2.8
                + 0.80 * click_label
                + 0.55 * item["merchant_quality"]
                + 0.42 * user["historical_cvr"]
                + 0.30 * category_match
                + 0.24 * semantic_score
                + 0.16 * min(item["price"] / 1500.0, 1.0)
                - 0.22 * price_gap
                + 0.08 * (request_category in {"finance", "education"})
            )
            cvr_probability = float(sigmoid(cvr_logit + rng.normal(0, 0.18)))
            conversion_label = int(click_label and rng.random() < cvr_probability)

            records.append(
                {
                    "request_id": f"req_{request_index:05d}",
                    "split": split,
                    "user_id": user["user_id"],
                    "item_id": item["item_id"],
                    "query_text": query_text,
                    "title_text": item["title_text"],
                    "combined_text": f"{query_text} [SEP] {item['title_text']}",
                    "query_category": request_category,
                    "item_category": item["item_category"],
                    "device": user["device"],
                    "city_tier": user["city_tier"],
                    "gender": user["gender"],
                    "hour": hour,
                    "is_weekend": is_weekend,
                    "user_age": user["user_age"],
                    "user_activity": user["user_activity"],
                    "historical_ctr": user["historical_ctr"],
                    "historical_cvr": user["historical_cvr"],
                    "price": item["price"],
                    "bid": item["bid"],
                    "merchant_quality": item["merchant_quality"],
                    "freshness": item["freshness"],
                    "discount_rate": item["discount_rate"],
                    "query_length": len(query_text.split()),
                    "title_length": len(item["title_text"].split()),
                    "token_overlap": overlap,
                    "category_match": category_match,
                    "price_gap": price_gap,
                    "semantic_score": semantic_score,
                    "position_index": float(position_index),
                    "click_label": click_label,
                    "conversion_label": conversion_label,
                }
            )

    dataframe = pd.DataFrame.from_records(records)
    dataframe.attrs["config"] = asdict(config)
    return dataframe
