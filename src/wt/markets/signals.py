"""Trade-signal generation from model probabilities and venue markets."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from wt.markets.ev import evaluate_contract


def _price(value: Any) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed / 100.0 if parsed > 1.0 else parsed


def _merge_key_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "venue",
        "ticker",
        "station",
        "target",
        "target_local_date",
        "bucket_lo",
        "bucket_hi",
    ]
    return [column for column in candidates if column in frame.columns]


def generate_signals(
    predictions: pd.DataFrame,
    markets: pd.DataFrame,
    min_edge_bps: int = 300,
    max_position_pct: float = 0.02,
    *,
    fee_rate: float = 0.0,
) -> pd.DataFrame:
    """Join bucket probabilities to market prices and return ranked +EV signals.

    ``predictions`` must contain ``model_prob`` and bucket columns. ``markets``
    must contain tradable prices, normally ``yes_ask`` and optionally
    ``no_ask``. The function is venue-agnostic; Kalshi and Polymarket rows can
    be mixed as long as their keys match.
    """

    if predictions.empty or markets.empty:
        return _empty_signals()

    prediction_rows = predictions.copy()
    market_rows = markets.copy()
    for frame in (prediction_rows, market_rows):
        if "target_local_date" in frame.columns:
            frame["target_local_date"] = pd.to_datetime(frame["target_local_date"]).dt.date

    join_cols = _merge_key_columns(prediction_rows)
    join_cols = [column for column in join_cols if column in market_rows.columns]
    if not join_cols:
        raise ValueError("No common key columns available to join predictions and markets")

    joined = prediction_rows.merge(market_rows, on=join_cols, how="inner", suffixes=("", "_market"))
    rows: list[dict[str, Any]] = []
    generated_at = datetime.now(tz=UTC)
    for record in joined.to_dict(orient="records"):
        yes_ask = _price(record.get("yes_ask"))
        if yes_ask is None:
            continue
        no_ask = _price(record.get("no_ask"))
        model_prob = float(record["model_prob"])
        ev = evaluate_contract(
            model_prob,
            yes_ask,
            no_ask,
            fee_rate=fee_rate,
            max_position_pct=max_position_pct,
        )
        if ev.edge_bps < min_edge_bps:
            continue
        rows.append(
            {
                "venue": record.get("venue"),
                "ticker": record.get("ticker"),
                "station": record.get("station"),
                "target": record.get("target"),
                "target_local_date": record.get("target_local_date"),
                "bucket_lo": record.get("bucket_lo"),
                "bucket_hi": record.get("bucket_hi"),
                "model_prob": model_prob,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "ev_yes": ev.ev_yes,
                "ev_no": ev.ev_no,
                "side": ev.side,
                "edge_bps": ev.edge_bps,
                "kelly_fraction": ev.kelly_fraction,
                "generated_at": generated_at,
            }
        )

    if not rows:
        return _empty_signals()
    out = pd.DataFrame(rows)
    out.sort_values(["edge_bps", "kelly_fraction"], ascending=[False, False], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "venue",
            "ticker",
            "station",
            "target",
            "target_local_date",
            "bucket_lo",
            "bucket_hi",
            "model_prob",
            "yes_ask",
            "no_ask",
            "ev_yes",
            "ev_no",
            "side",
            "edge_bps",
            "kelly_fraction",
            "generated_at",
        ]
    )
