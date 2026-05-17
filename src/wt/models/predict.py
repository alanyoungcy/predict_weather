"""Probabilistic model inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, inf, isfinite, sqrt
from typing import Protocol

import numpy as np
import pandas as pd

from wt.models.train import TrainedModel


_MIN_SIGMA = 0.1


class Distribution(Protocol):
    """Common protocol for forecast distributions."""

    point: float

    def cdf(self, x: float) -> float:
        """Return P(X <= x)."""

    def pmf_bucket(self, lo: float, hi: float) -> float:
        """Return P(lo <= X < hi)."""


def _normal_cdf(x: float, mean: float, sigma: float) -> float:
    if x == -inf:
        return 0.0
    if x == inf:
        return 1.0
    sigma = max(float(sigma), _MIN_SIGMA)
    z = (float(x) - float(mean)) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


@dataclass(frozen=True, slots=True)
class NormalDistribution:
    """Parametric normal fallback distribution."""

    point: float
    sigma: float

    def cdf(self, x: float) -> float:
        return _normal_cdf(x, self.point, self.sigma)

    def pmf_bucket(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, self.cdf(hi) - self.cdf(lo)))


@dataclass(frozen=True, slots=True)
class QuantileDistribution:
    """Distribution defined by quantile forecasts with normal tail extrapolation."""

    point: float
    quantiles: dict[float, float]
    sigma: float

    def __post_init__(self) -> None:
        if not self.quantiles:
            raise ValueError("quantiles cannot be empty")
        valid = {
            float(prob): float(value)
            for prob, value in self.quantiles.items()
            if 0.0 < float(prob) < 1.0 and isfinite(float(value))
        }
        if not valid:
            raise ValueError("quantiles must contain finite values with probabilities in (0, 1)")
        ordered = dict(sorted(valid.items()))
        object.__setattr__(self, "quantiles", ordered)
        object.__setattr__(self, "sigma", max(float(self.sigma), _MIN_SIGMA))

    @property
    def _pairs(self) -> list[tuple[float, float]]:
        pairs = list(self.quantiles.items())
        monotone: list[tuple[float, float]] = []
        last_value = -inf
        for prob, value in pairs:
            value = max(value, last_value)
            monotone.append((prob, value))
            last_value = value
        return monotone

    def cdf(self, x: float) -> float:
        if x == -inf:
            return 0.0
        if x == inf:
            return 1.0

        x = float(x)
        pairs = self._pairs
        low_prob, low_value = pairs[0]
        high_prob, high_value = pairs[-1]

        if x <= low_value:
            # Scale the left normal tail so cdf(low_value) equals low_prob.
            return max(0.0, min(low_prob, low_prob * _normal_cdf(x, low_value, self.sigma) / 0.5))

        if x >= high_value:
            tail = (_normal_cdf(x, high_value, self.sigma) - 0.5) / 0.5
            return max(high_prob, min(1.0, high_prob + (1.0 - high_prob) * tail))

        for (p0, v0), (p1, v1) in zip(pairs, pairs[1:], strict=False):
            if v0 <= x <= v1:
                if v1 == v0:
                    return max(p0, min(p1, (p0 + p1) / 2.0))
                weight = (x - v0) / (v1 - v0)
                return max(0.0, min(1.0, p0 + weight * (p1 - p0)))

        return max(0.0, min(1.0, _normal_cdf(x, self.point, self.sigma)))

    def pmf_bucket(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, self.cdf(hi) - self.cdf(lo)))


def _feature_frame(model: TrainedModel, features_row: dict | pd.Series) -> pd.DataFrame:
    row = dict(features_row)
    missing = [name for name in model.feature_names if name not in row]
    if missing:
        raise ValueError(f"Missing model feature columns: {', '.join(missing)}")
    return pd.DataFrame([{name: row[name] for name in model.feature_names}])


def _predict_scalar(estimator: object, frame: pd.DataFrame) -> float:
    predicted = estimator.predict(frame)  # type: ignore[attr-defined]
    return float(np.asarray(predicted, dtype="float64").ravel()[0])


def predict_distribution(model: TrainedModel, features_row: dict | pd.Series) -> Distribution:
    """Return a calibrated forecast distribution for one feature row.

    Quantile heads are preferred. If they are unavailable or fail at inference,
    the function returns a normal distribution centered on the point model.
    """

    frame = _feature_frame(model, features_row)
    point = _predict_scalar(model.point_model, frame)
    sigma = max(float(model.residual_sigma or 0.0), _MIN_SIGMA)

    quantiles: dict[float, float] = {}
    for prob, estimator in sorted(model.quantile_models.items()):
        try:
            quantiles[float(prob)] = _predict_scalar(estimator, frame)
        except Exception:
            quantiles = {}
            break

    if quantiles:
        return QuantileDistribution(point=point, quantiles=quantiles, sigma=sigma)
    return NormalDistribution(point=point, sigma=sigma)


def bucket_probabilities(dist: Distribution, buckets: list[tuple[float, float]]) -> list[float]:
    """Return normalized probabilities for an ordered bucket list."""

    raw = [dist.pmf_bucket(float(lo), float(hi)) for lo, hi in buckets]
    total = float(sum(raw))
    if total <= 0.0:
        if not raw:
            return []
        return [1.0 / len(raw)] * len(raw)
    normalized = [prob / total for prob in raw]
    # Keep exact sum stable for downstream tests and reports.
    normalized[-1] += 1.0 - sum(normalized)
    return normalized


def distribution_from_prediction_record(record: dict | pd.Series) -> Distribution:
    """Reconstruct a distribution from a persisted prediction row."""

    row = dict(record)
    point = float(row["point_pred"])
    sigma = max(float(row.get("sigma") or _MIN_SIGMA), _MIN_SIGMA)
    quantiles = {
        0.05: row.get("q05"),
        0.25: row.get("q25"),
        0.5: row.get("q50"),
        0.75: row.get("q75"),
        0.95: row.get("q95"),
    }
    clean_quantiles = {
        prob: float(value)
        for prob, value in quantiles.items()
        if value is not None and pd.notna(value)
    }
    if len(clean_quantiles) >= 2 and row.get("dist_family") != "normal":
        return QuantileDistribution(point=point, quantiles=clean_quantiles, sigma=sigma)
    return NormalDistribution(point=point, sigma=sigma)
