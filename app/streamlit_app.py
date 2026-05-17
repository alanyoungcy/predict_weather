"""Streamlit dashboard for training status and venue trading signals."""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wt.dashboard.data import (
    DashboardArtifacts,
    filter_frame,
    load_dashboard_artifacts,
    station_options,
    summarize_labels,
)


st.set_page_config(
    page_title="Weather Trader",
    page_icon="WT",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1500px;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    .wt-hero {
      padding: 1.1rem 1.25rem;
      border-radius: 20px;
      background:
        radial-gradient(circle at 10% 20%, rgba(255, 190, 92, .42), transparent 28%),
        linear-gradient(135deg, #102820 0%, #1D4E3B 42%, #D7B46A 140%);
      color: #fffaf0;
      border: 1px solid rgba(255,255,255,.22);
    }
    .wt-muted {color: #667085; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)
def _load() -> DashboardArtifacts:
    return load_dashboard_artifacts()


def _metric_count(frame: pd.DataFrame) -> int:
    return 0 if frame.empty else len(frame)


def _safe_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(pd.Timestamp(value).date())


def _run_worker_command(command: list[str], timeout: int = 180) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return completed.returncode, output.strip()


def _download_button(label: str, frame: pd.DataFrame, filename: str) -> None:
    if frame.empty:
        return
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


artifacts = _load()
stations = station_options()

with st.sidebar:
    st.header("Controls")
    selected_stations = st.multiselect("Stations", stations, default=stations[:4])
    selected_venues = st.multiselect(
        "Venues",
        ["kalshi", "polymarket"],
        default=["kalshi", "polymarket"],
    )
    selected_targets = st.multiselect(
        "Targets",
        ["tmax", "tmin", "apcp24"],
        default=["tmax", "tmin", "apcp24"],
    )
    min_edge = st.slider("Min edge (bps)", min_value=0, max_value=3000, value=300, step=50)
    st.divider()
    if st.button("Refresh artifacts"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Worker Actions")
    dry_station = st.selectbox(
        "Dry-run station",
        stations,
        index=stations.index("KNYC") if "KNYC" in stations else 0,
    )
    dry_date = st.date_input("Target date", value=date.today() + timedelta(days=1))
    if st.button("Run forecast dry-run"):
        command = [
            "/opt/miniconda3/envs/agentenv/bin/python",
            "-m",
            "wt.orchestration.cron_evening",
            "--dry-run",
            "--station",
            dry_station,
            "--target-date",
            dry_date.isoformat(),
        ]
        with st.spinner("Running worker dry-run..."):
            try:
                code, output = _run_worker_command(command)
                st.code(output or f"exit={code}", language="text")
            except subprocess.TimeoutExpired:
                st.error("Dry-run timed out after 180 seconds.")

st.markdown(
    """
    <div class="wt-hero">
      <h1 style="margin:0 0 .25rem 0;">Weather Trader Control Room</h1>
      <div>Training health, forecast distributions, and read-only venue signal monitoring.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

filtered_predictions = filter_frame(
    artifacts.predictions,
    stations=selected_stations,
    targets=selected_targets,
)
filtered_bucketed = filter_frame(
    artifacts.bucketed_predictions,
    stations=selected_stations,
    venues=selected_venues,
    targets=selected_targets,
)
filtered_markets = filter_frame(
    artifacts.markets,
    stations=selected_stations,
    venues=selected_venues,
    targets=selected_targets,
)
filtered_signals = filter_frame(
    artifacts.signals,
    stations=selected_stations,
    venues=selected_venues,
    targets=selected_targets,
)
if not filtered_signals.empty and "edge_bps" in filtered_signals.columns:
    filtered_signals = filtered_signals[filtered_signals["edge_bps"] >= min_edge]

overview, training, forecasts, trading, health = st.tabs(
    ["Overview", "Training", "Forecasts", "Trading", "Data Health"]
)

with overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Labels", f"{_metric_count(artifacts.labels):,}")
    c2.metric("Models", f"{_metric_count(artifacts.model_inventory):,}")
    c3.metric("Predictions", f"{_metric_count(filtered_predictions):,}")
    c4.metric("Signals", f"{_metric_count(filtered_signals):,}")
    top_edge = (
        int(filtered_signals["edge_bps"].max())
        if not filtered_signals.empty and "edge_bps" in filtered_signals
        else 0
    )
    c5.metric("Top Edge", f"{top_edge:,} bps")

    st.subheader("Latest Signals")
    if filtered_signals.empty:
        st.info(
            "No signal parquet exists yet. Run live prediction with `--with-markets` "
            "after models are trained."
        )
    else:
        columns = [
            col
            for col in [
                "venue",
                "ticker",
                "station",
                "target",
                "target_local_date",
                "side",
                "model_prob",
                "yes_ask",
                "no_ask",
                "edge_bps",
                "kelly_fraction",
            ]
            if col in filtered_signals.columns
        ]
        st.dataframe(filtered_signals[columns], use_container_width=True, hide_index=True)

    label_summary = summarize_labels(artifacts.labels)
    if not label_summary.empty:
        st.subheader("Ground Truth Coverage")
        st.bar_chart(label_summary.set_index("station")["days"])

with training:
    st.subheader("Model Inventory")
    if artifacts.model_inventory.empty:
        st.warning(
            "No model files found under `models/`. "
            "Train models after historical features are built."
        )
    else:
        st.dataframe(artifacts.model_inventory, use_container_width=True, hide_index=True)
        st.bar_chart(artifacts.model_inventory.groupby(["station", "target"]).size())

    st.subheader("Training Metrics")
    if artifacts.metrics.empty:
        st.info("No `metrics.json` or `metrics_*.json` files found yet.")
    else:
        st.dataframe(artifacts.metrics, use_container_width=True, hide_index=True)
        numeric_cols = artifacts.metrics.select_dtypes("number").columns.tolist()
        if "test_mae" in numeric_cols:
            st.bar_chart(artifacts.metrics.set_index(["station", "target"])["test_mae"])
        _download_button("Download metrics CSV", artifacts.metrics, "training_metrics.csv")

    st.subheader("Feature Importance")
    if artifacts.feature_importance.empty:
        st.info("No feature-importance CSV exists yet.")
    else:
        top_features = artifacts.feature_importance.sort_values(
            "importance",
            ascending=False,
        ).head(30)
        st.dataframe(top_features, use_container_width=True, hide_index=True)
        if {"feature", "importance"}.issubset(top_features.columns):
            st.bar_chart(top_features.set_index("feature")["importance"])

    st.subheader("Training Commands")
    st.code(
        "\n".join(
            [
                "/opt/miniconda3/envs/agentenv/bin/python "
                "scripts/bootstrap_historical_features.py "
                "--start 2018-01-01 --end 2026-05-16 --station all",
                "/opt/miniconda3/envs/agentenv/bin/python scripts/train_all.py "
                "--stations all --targets all --output models/v$(date +%Y%m%d) "
                "--promote-current",
            ]
        ),
        language="bash",
    )

with forecasts:
    st.subheader("Point and Quantile Predictions")
    if filtered_predictions.empty:
        st.info("No prediction parquet exists yet.")
    else:
        st.dataframe(filtered_predictions, use_container_width=True, hide_index=True)
        required_columns = {"target_local_date", "point_pred", "station", "target"}
        if required_columns.issubset(filtered_predictions.columns):
            chart_frame = filtered_predictions.copy()
            chart_frame["series"] = chart_frame["station"] + " " + chart_frame["target"]
            chart_frame["target_local_date"] = pd.to_datetime(
                chart_frame["target_local_date"]
            )
            st.line_chart(chart_frame, x="target_local_date", y="point_pred", color="series")
        _download_button("Download predictions CSV", filtered_predictions, "predictions.csv")

    st.subheader("Bucketed Probabilities")
    if filtered_bucketed.empty:
        st.info("No bucketed-probability parquet exists yet.")
    else:
        st.dataframe(filtered_bucketed, use_container_width=True, hide_index=True)
        _download_button("Download bucket probabilities CSV", filtered_bucketed, "bucket_probs.csv")

with trading:
    st.subheader("Read-Only Trading Signals")
    st.caption("This dashboard never places orders. It only displays calculated +EV signals.")
    if filtered_signals.empty:
        st.warning("No signals match the current filters.")
    else:
        st.dataframe(filtered_signals, use_container_width=True, hide_index=True)
        if {"ticker", "edge_bps"}.issubset(filtered_signals.columns):
            signal_chart = filtered_signals.sort_values("edge_bps", ascending=False).head(20)
            st.bar_chart(signal_chart.set_index("ticker")["edge_bps"])
        _download_button("Download signals CSV", filtered_signals, "signals.csv")

    st.subheader("Market Snapshot")
    if filtered_markets.empty:
        st.info("No market snapshot parquet exists yet.")
    else:
        st.dataframe(filtered_markets, use_container_width=True, hide_index=True)

    st.subheader("Signal Generation Command")
    st.code(
        "/opt/miniconda3/envs/agentenv/bin/python -m wt.orchestration.cron_evening "
        "--station KNYC --with-markets --min-edge-bps 300",
        language="bash",
    )

with health:
    st.subheader("Run Summaries")
    if artifacts.runs.empty:
        st.info("No run summary parquet exists yet.")
    else:
        st.dataframe(
            artifacts.runs.sort_values("generated_at", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Label Coverage")
    label_summary = summarize_labels(artifacts.labels)
    if label_summary.empty:
        st.warning("No labels found.")
    else:
        label_summary["start"] = label_summary["start"].map(_safe_date)
        label_summary["end"] = label_summary["end"].map(_safe_date)
        st.dataframe(label_summary, use_container_width=True, hide_index=True)

    st.subheader("Artifact Locations")
    st.code(
        """
data/labels/labels.parquet
data/features/live/
data/predictions/
data/predictions/bucketed/
data/markets/
data/signals/
data/runs/
models/
        """.strip(),
        language="text",
    )
