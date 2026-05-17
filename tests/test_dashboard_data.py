import json
from pathlib import Path

import pandas as pd

from wt.dashboard.data import (
    filter_frame,
    load_feature_importance,
    load_model_inventory,
    load_model_metrics,
    read_parquet_collection,
    summarize_labels,
)


def test_read_parquet_collection_combines_files(tmp_path: Path):
    pd.DataFrame([{"x": 1}]).to_parquet(tmp_path / "a.parquet")
    pd.DataFrame([{"x": 2}]).to_parquet(tmp_path / "b.parquet")
    frame = read_parquet_collection(tmp_path)
    assert sorted(frame["x"].tolist()) == [1, 2]


def test_load_model_inventory_and_metrics(tmp_path: Path):
    version = tmp_path / "v20260517"
    version.mkdir()
    (version / "model_KNYC_tmax.pkl").write_bytes(b"fake")
    (version / "metrics.json").write_text(
        json.dumps({"KNYC_tmax": {"test_mae": 1.2}}),
        encoding="utf-8",
    )
    pd.DataFrame([{"station": "KNYC", "target": "tmax", "feature": "x", "importance": 1.0}]).to_csv(
        version / "feature_importance.csv",
        index=False,
    )
    assert len(load_model_inventory(tmp_path)) == 1
    assert load_model_metrics(tmp_path).loc[0, "test_mae"] == 1.2
    assert load_feature_importance(tmp_path).loc[0, "feature"] == "x"


def test_summarize_labels_and_filter_frame():
    labels = pd.DataFrame(
        [
            {
                "station": "KNYC",
                "local_date": "2026-01-01",
                "tmax_f": 40,
                "tmin_f": 30,
                "precip_in": 0.0,
            },
            {
                "station": "KNYC",
                "local_date": "2026-01-02",
                "tmax_f": 42,
                "tmin_f": 31,
                "precip_in": 0.2,
            },
        ]
    )
    summary = summarize_labels(labels)
    assert summary.loc[0, "days"] == 2
    assert summary.loc[0, "wet_days"] == 1

    frame = pd.DataFrame(
        [
            {"station": "KNYC", "venue": "kalshi", "target": "tmax"},
            {"station": "KBOS", "venue": "polymarket", "target": "tmin"},
        ]
    )
    filtered = filter_frame(frame, stations=["KNYC"], venues=["kalshi"], targets=["tmax"])
    assert len(filtered) == 1
