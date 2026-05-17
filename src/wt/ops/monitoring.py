"""Run summaries and model drift detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class DriftAlert:
    station: str
    target: str
    metric: str
    value: float
    threshold: float


TARGET_COLUMNS = {"tmax": "tmax_f", "tmin": "tmin_f", "apcp24": "precip_in"}


def build_run_summary(
    *,
    run_id: str,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Create one observability row for a pipeline run."""

    top_edge = float(signals["edge_bps"].max()) if not signals.empty else np.nan
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "feature_rows": int(len(features)),
                "prediction_rows": int(len(predictions)),
                "signal_rows": int(len(signals)),
                "top_signal_edge_bps": top_edge,
                "generated_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )


def compute_residuals(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join predictions to labels and compute prediction residuals."""

    if predictions.empty or labels.empty:
        return pd.DataFrame(
            columns=["station", "target", "target_local_date", "actual", "point_pred", "residual"]
        )

    labels = labels.copy()
    labels["local_date"] = pd.to_datetime(labels["local_date"]).dt.date
    preds = predictions.copy()
    preds["target_local_date"] = pd.to_datetime(preds["target_local_date"]).dt.date

    rows: list[pd.DataFrame] = []
    for target, label_col in TARGET_COLUMNS.items():
        subset = preds[preds["target"] == target]
        if subset.empty:
            continue
        joined = subset.merge(
            labels[["station", "local_date", label_col]],
            left_on=["station", "target_local_date"],
            right_on=["station", "local_date"],
            how="inner",
        )
        if joined.empty:
            continue
        out = joined[["station", "target", "target_local_date", "point_pred"]].copy()
        out["actual"] = joined[label_col].astype("float64")
        out["residual"] = out["point_pred"].astype("float64") - out["actual"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def detect_drift(
    residuals: pd.DataFrame,
    baseline_mae: dict[tuple[str, str], float],
    *,
    window: int = 14,
    mae_multiplier: float = 1.5,
    bias_threshold: float = 1.0,
) -> list[DriftAlert]:
    """Return alerts when rolling MAE or bias exceeds configured thresholds."""

    if residuals.empty:
        return []
    frame = residuals.copy()
    frame["target_local_date"] = pd.to_datetime(frame["target_local_date"])
    alerts: list[DriftAlert] = []
    for (station, target), group in frame.groupby(["station", "target"]):
        group = group.sort_values("target_local_date").tail(window)
        if group.empty:
            continue
        rolling_mae = float(group["residual"].abs().mean())
        rolling_bias = float(group["residual"].mean())
        base = float(baseline_mae.get((station, target), np.nan))
        if not np.isnan(base) and rolling_mae > base * mae_multiplier:
            alerts.append(
                DriftAlert(
                    station=station,
                    target=target,
                    metric="rolling_mae",
                    value=rolling_mae,
                    threshold=base * mae_multiplier,
                )
            )
        if abs(rolling_bias) > bias_threshold:
            alerts.append(
                DriftAlert(
                    station=station,
                    target=target,
                    metric="rolling_bias",
                    value=rolling_bias,
                    threshold=bias_threshold,
                )
            )
    return alerts
