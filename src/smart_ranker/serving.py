from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from .model import DeepModelArtifacts, predict_deep_model
from .ranking import build_ranked_candidates


class CandidateRecord(BaseModel):
    request_id: str
    user_id: str | None = None
    item_id: str
    query_text: str
    title_text: str
    query_category: str
    item_category: str
    device: str
    city_tier: str
    gender: str
    hour: int
    is_weekend: int
    user_age: float
    user_activity: float
    historical_ctr: float
    historical_cvr: float
    price: float
    bid: float
    merchant_quality: float
    freshness: float
    discount_rate: float
    token_overlap: float
    category_match: float
    price_gap: float
    semantic_score: float
    position_index: float
    query_length: int | None = None
    title_length: int | None = None
    combined_text: str | None = None
    click_label: int = 0
    conversion_label: int = 0


class RankRequest(BaseModel):
    candidates: list[CandidateRecord]


def _prepare_frame(records: list[CandidateRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = record.model_dump()
        payload["query_length"] = payload["query_length"] or len(payload["query_text"].split())
        payload["title_length"] = payload["title_length"] or len(payload["title_text"].split())
        payload["combined_text"] = payload["combined_text"] or (
            f"{payload['query_text']} [SEP] {payload['title_text']}"
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def create_app(artifact_path: str) -> FastAPI:
    artifacts = DeepModelArtifacts.load(artifact_path)
    app = FastAPI(title="Smart Ranker API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/rank")
    def rank(request: RankRequest) -> dict[str, Any]:
        frame = _prepare_frame(request.candidates)
        ctr_scores, cvr_scores, _ = predict_deep_model(artifacts, frame)
        ranked = build_ranked_candidates(frame, ctr_scores, cvr_scores)
        return {
            "results": ranked[
                ["request_id", "item_id", "pred_ctr", "pred_cvr", "business_score", "query_category", "item_category"]
            ].to_dict(orient="records")
        }

    return app
