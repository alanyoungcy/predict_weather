"""Market bucket parsing and probability alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import inf
from typing import Any

import pandas as pd

from wt.models.predict import (
    Distribution,
    bucket_probabilities as distribution_bucket_probabilities,
    distribution_from_prediction_record,
)


@dataclass(frozen=True, slots=True)
class Bucket:
    """One tradable outcome bucket from a venue market."""

    ticker: str
    lo: float
    hi: float
    venue: str = "kalshi"
    market_id: str | None = None
    title: str | None = None
    station: str | None = None
    target: str | None = None
    yes_ask: float | None = None
    no_ask: float | None = None


_NUMBER = r"(-?\d+(?:\.\d+)?)"
_DASH_RE = re.compile(rf"\b{_NUMBER}\s*(?:-|to|through|thru|and)\s*{_NUMBER}\b", re.IGNORECASE)
_BETWEEN_RE = re.compile(rf"\bbetween\s+{_NUMBER}\s+(?:and|to|-)\s+{_NUMBER}\b", re.IGNORECASE)
_BELOW_RE = re.compile(
    rf"\b(?:below|under|less than|lower than|at most|no more than)\s+{_NUMBER}\b",
    re.IGNORECASE,
)
_ABOVE_RE = re.compile(
    rf"\b(?:above|over|greater than|more than|higher than|at least|no less than)\s+{_NUMBER}\b",
    re.IGNORECASE,
)
_OR_BELOW_RE = re.compile(
    rf"\b{_NUMBER}\s*(?:or below|or lower|and below|and lower)\b",
    re.IGNORECASE,
)
_OR_ABOVE_RE = re.compile(
    rf"\b{_NUMBER}\s*(?:or above|or higher|and above|and higher)\b",
    re.IGNORECASE,
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_bucket_range(text: str) -> tuple[float, float] | None:
    """Parse common weather contract bucket wording into ``[lo, hi)`` bounds."""

    normalized = text.replace("°", " ").replace("$", " ")
    match = _BETWEEN_RE.search(normalized) or _DASH_RE.search(normalized)
    if match:
        lo, hi = sorted((float(match.group(1)), float(match.group(2))))
        return lo, hi

    match = _BELOW_RE.search(normalized) or _OR_BELOW_RE.search(normalized)
    if match:
        return -inf, float(match.group(1))

    match = _ABOVE_RE.search(normalized) or _OR_ABOVE_RE.search(normalized)
    if match:
        return float(match.group(1)), inf

    return None


def _bucket_from_payload(
    payload: Any,
    *,
    venue: str,
    default_station: str | None,
    default_target: str | None,
) -> Bucket | None:
    text_parts = [
        str(_get(payload, key, "") or "")
        for key in ("title", "subtitle", "rules_primary", "rules_secondary", "name", "question")
    ]
    text = " ".join(part for part in text_parts if part)
    parsed = parse_bucket_range(text)
    if parsed is None:
        lo = _get(payload, "bucket_lo", _get(payload, "lo"))
        hi = _get(payload, "bucket_hi", _get(payload, "hi"))
        if lo is None or hi is None:
            return None
        parsed = (float(lo), float(hi))

    return Bucket(
        ticker=str(
            _get(payload, "ticker", _get(payload, "token_id", _get(payload, "id", ""))) or ""
        ),
        market_id=str(_get(payload, "market_id", _get(payload, "condition_id", "")) or "") or None,
        lo=float(parsed[0]),
        hi=float(parsed[1]),
        venue=venue,
        title=text.strip() or None,
        station=_get(payload, "station", default_station),
        target=_get(payload, "target", default_target),
        yes_ask=_coerce_price(_get(payload, "yes_ask", _get(payload, "ask"))),
        no_ask=_coerce_price(_get(payload, "no_ask")),
    )


def _coerce_price(value: Any) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed / 100.0 if parsed > 1.0 else parsed


def extract_buckets_from_market(
    market: Any,
    *,
    venue: str = "kalshi",
    station: str | None = None,
    target: str | None = None,
) -> list[Bucket]:
    """Extract weather outcome buckets from a Kalshi or Polymarket payload."""

    explicit = _get(market, "buckets")
    if explicit:
        buckets = [
            bucket
            for item in explicit
            if (
                bucket := _bucket_from_payload(
                    item,
                    venue=venue,
                    default_station=station,
                    default_target=target,
                )
            )
        ]
        return sorted(buckets, key=lambda item: item.lo)

    outcomes = _get(market, "outcomes") or _get(market, "tokens") or []
    if outcomes:
        buckets = []
        for outcome in outcomes:
            merged = dict(market) if isinstance(market, dict) else {}
            if isinstance(outcome, dict):
                merged.update(outcome)
            else:
                merged["title"] = str(outcome)
            if bucket := _bucket_from_payload(
                merged,
                venue=venue,
                default_station=station,
                default_target=target,
            ):
                buckets.append(bucket)
        return sorted(buckets, key=lambda item: item.lo)

    bucket = _bucket_from_payload(
        market,
        venue=venue,
        default_station=station,
        default_target=target,
    )
    return [bucket] if bucket else []


def bucket_probabilities(dist: Distribution, buckets: list[Bucket]) -> list[float]:
    """Return normalized probabilities aligned to parsed market buckets."""

    return distribution_bucket_probabilities(dist, [(bucket.lo, bucket.hi) for bucket in buckets])


def align_predictions_to_buckets(
    predictions: pd.DataFrame,
    market_buckets: pd.DataFrame,
) -> pd.DataFrame:
    """Create the bucketed model-probability table used by signal generation."""

    if predictions.empty or market_buckets.empty:
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
            ]
        )

    bucket_rows = market_buckets.copy()
    bucket_rows["target_local_date"] = pd.to_datetime(bucket_rows["target_local_date"]).dt.date
    pred_rows = predictions.copy()
    pred_rows["target_local_date"] = pd.to_datetime(pred_rows["target_local_date"]).dt.date

    rows: list[dict[str, Any]] = []
    for key, group in bucket_rows.groupby(["venue", "station", "target", "target_local_date"]):
        venue, station, target, target_date = key
        pred_match = pred_rows[
            (pred_rows["station"] == station)
            & (pred_rows["target"] == target)
            & (pred_rows["target_local_date"] == target_date)
        ]
        if pred_match.empty:
            continue
        dist = distribution_from_prediction_record(pred_match.iloc[0])
        buckets = [
            Bucket(
                venue=str(row["venue"]),
                ticker=str(row["ticker"]),
                station=str(row["station"]),
                target=str(row["target"]),
                lo=float(row["bucket_lo"]),
                hi=float(row["bucket_hi"]),
            )
            for row in group.to_dict(orient="records")
        ]
        probs = bucket_probabilities(dist, buckets)
        for bucket, model_prob in zip(buckets, probs, strict=True):
            rows.append(
                {
                    "venue": venue,
                    "ticker": bucket.ticker,
                    "station": station,
                    "target": target,
                    "target_local_date": target_date,
                    "bucket_lo": bucket.lo,
                    "bucket_hi": bucket.hi,
                    "model_prob": model_prob,
                }
            )

    return pd.DataFrame(rows)
