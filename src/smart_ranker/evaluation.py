from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .config import CLICK_LABEL, CONVERSION_LABEL


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-5, 1 - 1e-5)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(log_loss(labels, clipped))


def ndcg_at_k(frame: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    working = pd.DataFrame(
        {
            "request_id": frame["request_id"].to_numpy(),
            "score": scores,
            "relevance": frame[CLICK_LABEL].to_numpy() + 4 * frame[CONVERSION_LABEL].to_numpy(),
        }
    )

    def dcg(values: np.ndarray) -> float:
        return float(
            np.sum([(2 ** value - 1) / math.log2(index + 2) for index, value in enumerate(values[:k])])
        )

    group_scores: list[float] = []
    for _, group in working.groupby("request_id"):
        ideal = np.sort(group["relevance"].to_numpy())[::-1]
        if ideal.max(initial=0) <= 0:
            continue
        ranked = group.sort_values("score", ascending=False)["relevance"].to_numpy()
        denom = dcg(ideal)
        if denom > 0:
            group_scores.append(dcg(ranked) / denom)
    return float(np.mean(group_scores)) if group_scores else 0.0


def evaluate_predictions(
    frame: pd.DataFrame,
    ctr_scores: np.ndarray,
    cvr_scores: np.ndarray,
    business_scores: np.ndarray,
) -> dict[str, float]:
    ctr_labels = frame[CLICK_LABEL].to_numpy()
    cvr_labels = frame[CONVERSION_LABEL].to_numpy()
    return {
        "ctr_auc": safe_auc(ctr_labels, ctr_scores),
        "ctr_logloss": safe_logloss(ctr_labels, ctr_scores),
        "cvr_auc": safe_auc(cvr_labels, cvr_scores),
        "cvr_logloss": safe_logloss(cvr_labels, cvr_scores),
        "ndcg_at_10": ndcg_at_k(frame, business_scores, k=10),
        "avg_pred_ctr": float(np.mean(ctr_scores)),
        "avg_pred_cvr": float(np.mean(cvr_scores)),
        "avg_business_score": float(np.mean(business_scores)),
    }
