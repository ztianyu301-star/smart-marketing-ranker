from __future__ import annotations

import numpy as np
import pandas as pd


def business_score(frame: pd.DataFrame, ctr_scores: np.ndarray, cvr_scores: np.ndarray) -> np.ndarray:
    normalized_price = np.clip(frame["price"].to_numpy(dtype=float) / 1000.0, 0.05, 10.0)
    bids = frame["bid"].to_numpy(dtype=float)
    return bids * ctr_scores + 0.7 * normalized_price * ctr_scores * cvr_scores


def build_ranked_candidates(
    frame: pd.DataFrame,
    ctr_scores: np.ndarray,
    cvr_scores: np.ndarray,
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["pred_ctr"] = ctr_scores
    ranked["pred_cvr"] = cvr_scores
    ranked["business_score"] = business_score(ranked, ctr_scores, cvr_scores)
    return ranked.sort_values(["request_id", "business_score"], ascending=[True, False]).reset_index(drop=True)
