"""Shared feature engineering transforms."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


TARGET_COLUMNS = ("tmax_f", "tmin_f", "precip_in")


def add_lag_features(df: pd.DataFrame, label_df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    lags = lags or [1, 3, 7, 14]
    base = df.copy()
    labels = label_df.copy().sort_values(["station", "local_date"])
    labels = labels[["station", "local_date", *TARGET_COLUMNS]]

    for lag in lags:
        shifted = labels.copy()
        shifted["target_local_date"] = pd.to_datetime(shifted["local_date"]) + pd.Timedelta(days=lag)
        rename_map = {
            "tmax_f": f"obs_tmax_lag{lag}_f",
            "tmin_f": f"obs_tmin_lag{lag}_f",
            "precip_in": f"obs_precip_lag{lag}_in",
        }
        shifted.rename(columns=rename_map, inplace=True)
        base = base.merge(
            shifted[["station", "target_local_date", *rename_map.values()]],
            on=["station", "target_local_date"],
            how="left",
        )

    if {"obs_tmax_lag1_f", "obs_tmax_lag7_f"}.issubset(base.columns):
        cols = [f"obs_tmax_lag{lag}_f" for lag in [1, 3, 7] if f"obs_tmax_lag{lag}_f" in base.columns]
        if cols:
            base["obs_tmax_lag7_mean_f"] = base[cols].mean(axis=1)
    if {"obs_precip_lag1_in", "obs_precip_lag7_in"}.issubset(base.columns):
        cols = [f"obs_precip_lag{lag}_in" for lag in [1, 3, 7] if f"obs_precip_lag{lag}_in" in base.columns]
        if cols:
            base["obs_apcp_lag7_sum_in"] = base[cols].sum(axis=1)
    return base


def add_doy_encoding(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    dates = pd.to_datetime(base["target_local_date"])
    doy = dates.dt.dayofyear.astype("float64")
    base["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    base["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    base["month"] = dates.dt.month.astype("int8")
    return base


def add_model_spread(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    tmax_cols = [col for col in base.columns if col.endswith("_t2m_max_f")]
    apcp_cols = [col for col in base.columns if col.endswith("_apcp_24h_in") or col.endswith("_qpf_24h_in")]
    if tmax_cols:
        base["model_spread_tmax_f"] = base[tmax_cols].std(axis=1, ddof=0)
    if apcp_cols:
        base["model_spread_apcp_in"] = base[apcp_cols].std(axis=1, ddof=0)
    return base


def build_climatology(labels: pd.DataFrame, window_days: int = 15) -> pd.DataFrame:
    frame = labels.copy()
    dates = pd.to_datetime(frame["local_date"])
    frame["day_of_year"] = dates.dt.dayofyear
    rows: list[dict[str, float | str | int]] = []
    for (station, doy), group in frame.groupby(["station", "day_of_year"]):
        lower = doy - window_days
        upper = doy + window_days
        mask = (frame["station"] == station) & frame["day_of_year"].between(lower, upper)
        window = frame.loc[mask]
        rows.append(
            {
                "station": station,
                "day_of_year": doy,
                "climo_tmax_f": float(window["tmax_f"].mean()),
                "climo_tmin_f": float(window["tmin_f"].mean()),
                "climo_precip_in": float(window["precip_in"].mean()),
            }
        )
    return pd.DataFrame(rows)


def add_climatology(df: pd.DataFrame, climo_df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["day_of_year"] = pd.to_datetime(base["target_local_date"]).dt.dayofyear
    merged = base.merge(climo_df, on=["station", "day_of_year"], how="left")
    return merged.drop(columns=["day_of_year"])


def add_persistence_features(df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    labels = label_df.copy()
    labels["month_day"] = pd.to_datetime(labels["local_date"]).dt.strftime("%m-%d")
    stats = (
        labels.groupby(["station", "month_day"])
        .agg(
            persistence_tmax_f=("tmax_f", "mean"),
            persistence_tmin_f=("tmin_f", "mean"),
            persistence_precip_in=("precip_in", "mean"),
        )
        .reset_index()
    )
    base["month_day"] = pd.to_datetime(base["target_local_date"]).dt.strftime("%m-%d")
    merged = base.merge(stats, on=["station", "month_day"], how="left")
    return merged.drop(columns=["month_day"])
