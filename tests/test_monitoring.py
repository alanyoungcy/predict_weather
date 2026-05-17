from datetime import date, timedelta

import pandas as pd

from wt.ops.monitoring import build_run_summary, compute_residuals, detect_drift


def test_build_run_summary_counts_rows():
    summary = build_run_summary(
        run_id="r1",
        features=pd.DataFrame([{"station": "KNYC"}]),
        predictions=pd.DataFrame([{"station": "KNYC"}]),
        signals=pd.DataFrame([{"edge_bps": 500}]),
    )
    assert summary.loc[0, "feature_rows"] == 1
    assert summary.loc[0, "top_signal_edge_bps"] == 500


def test_compute_residuals_and_detect_drift():
    start = date(2026, 1, 1)
    predictions = pd.DataFrame(
        [
            {
                "station": "KNYC",
                "target": "tmax",
                "target_local_date": start + timedelta(days=idx),
                "point_pred": 75.0,
            }
            for idx in range(14)
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "station": "KNYC",
                "local_date": start + timedelta(days=idx),
                "tmax_f": 70.0,
                "tmin_f": 60.0,
                "precip_in": 0.0,
            }
            for idx in range(14)
        ]
    )
    residuals = compute_residuals(predictions, labels)
    alerts = detect_drift(residuals, {("KNYC", "tmax"): 2.0})
    assert len(residuals) == 14
    assert {alert.metric for alert in alerts} == {"rolling_mae", "rolling_bias"}
