"""Herbie-backed NWP point forecast extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import xarray as xr

from wt.config import AppSettings, load_settings


@dataclass(frozen=True, slots=True)
class VariableSpec:
    request: str
    search: str
    unit: str
    output_name: str


_VARIABLE_SPECS: dict[str, VariableSpec] = {
    "TMP:2 m": VariableSpec("TMP:2 m", "TMP:2 m", "kelvin", "t2m_f"),
    "APCP:surface": VariableSpec("APCP:surface", "APCP:surface", "kg_m2", "apcp_in"),
}

_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "hrrr": {"product": "sfc"},
    "gfs": {"product": "pgrb2.0p25"},
    "gefs": {"product": "atmos.0p50"},
    "nbm": {"product": "core"},
    "aifs": {"product": "single-levels"},
}


def _load_herbie_class():
    from herbie import Herbie

    return Herbie


def _ensure_herbie_env(settings: AppSettings) -> None:
    os.environ.setdefault("HERBIE_SAVE_DIR", str(settings.herbie_save_dir))
    os.environ.setdefault("HERBIE_CONFIG_PATH", str(settings.herbie_config_path))
    settings.herbie_save_dir.mkdir(parents=True, exist_ok=True)
    settings.herbie_config_path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_init_run(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return ts


def _spec_for(variable: str) -> VariableSpec:
    return _VARIABLE_SPECS.get(variable, VariableSpec(variable, variable, "raw", variable.lower().replace(":", "_")))


def _first_data_var(ds: xr.Dataset) -> xr.DataArray:
    for name, data_array in ds.data_vars.items():
        if name.lower() not in {"gribfile_projection", "lambert_conformal_conic"}:
            return data_array
    raise ValueError("Dataset contains no usable data variables")


def _extract_valid_time(ds: xr.Dataset, init_run_utc: pd.Timestamp, forecast_hour: int) -> pd.Timestamp:
    for coord_name in ("valid_time", "time"):
        if coord_name in ds.coords:
            value = ds.coords[coord_name].values
            ts = pd.Timestamp(np.ravel(value)[0])
            if ts.tzinfo is None:
                ts = ts.tz_localize(UTC)
            else:
                ts = ts.tz_convert(UTC)
            return ts
    return init_run_utc + pd.Timedelta(hours=forecast_hour)


def _find_coord(ds: xr.Dataset, candidates: list[str]) -> xr.DataArray | None:
    for name in candidates:
        if name in ds.coords:
            return ds.coords[name]
        if name in ds:
            value = ds[name]
            if isinstance(value, xr.DataArray):
                return value
    return None


def _point_value_from_dataarray(data_array: xr.DataArray, station_lat: float, station_lon: float) -> float:
    lat_coord = _find_coord(data_array.to_dataset(name="var"), ["latitude", "lat"])
    lon_coord = _find_coord(data_array.to_dataset(name="var"), ["longitude", "lon"])
    if lat_coord is None or lon_coord is None:
        squeezed = data_array.squeeze(drop=True)
        return float(np.asarray(squeezed.values).ravel()[0])

    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        if lat_coord.name in data_array.dims and lon_coord.name in data_array.dims:
            point = data_array.interp({lat_coord.name: station_lat, lon_coord.name: station_lon}, method="linear")
            return float(np.asarray(point.squeeze(drop=True).values).ravel()[0])
        if lat_coord.dims == lon_coord.dims:
            lat_values = np.asarray(lat_coord.values, dtype="float64")
            lon_values = np.asarray(lon_coord.values, dtype="float64")
            nearest_idx = int(np.nanargmin((lat_values - station_lat) ** 2 + (lon_values - station_lon) ** 2))
            indexers = {lat_coord.dims[0]: nearest_idx}
            point = data_array.isel(indexers)
            return float(np.asarray(point.squeeze(drop=True).values).ravel()[0])

    lat_values = np.asarray(lat_coord.values, dtype="float64")
    lon_values = np.asarray(lon_coord.values, dtype="float64")
    distance = (lat_values - station_lat) ** 2 + (lon_values - station_lon) ** 2
    nearest_flat = int(np.nanargmin(distance))
    nearest_index = np.unravel_index(nearest_flat, distance.shape)
    indexers = {dim: idx for dim, idx in zip(lat_coord.dims, nearest_index, strict=True)}
    point = data_array.isel(indexers)
    return float(np.asarray(point.squeeze(drop=True).values).ravel()[0])


def _convert_value(value: float, unit: str) -> float:
    if unit == "kelvin":
        return ((value - 273.15) * 9.0 / 5.0) + 32.0
    if unit == "kg_m2":
        return value * 0.0393701
    return value


def _herbie_kwargs_for(model: str, settings: AppSettings) -> dict[str, Any]:
    defaults = _MODEL_DEFAULTS.get(model)
    if defaults is None:
        raise ValueError(f"Unsupported model {model!r}")
    return {
        "model": model,
        "product": defaults["product"],
        "save_dir": Path(settings.herbie_save_dir),
        "verbose": False,
    }


def get_model_point_forecast(
    model: str,
    init_run_utc: datetime | str | pd.Timestamp,
    forecast_hours: list[int],
    station_lat: float,
    station_lon: float,
    variables: list[str],
    *,
    settings: AppSettings | None = None,
) -> pd.DataFrame:
    """Fetch a point forecast time series for selected model variables.

    Returns columns:
    [`valid_time_utc`, `var_name`, `value`, `model`, `init_run_utc`, `forecast_hour`]
    """

    cfg = settings or load_settings()
    _ensure_herbie_env(cfg)
    Herbie = _load_herbie_class()
    init_ts = _normalize_init_run(init_run_utc)

    rows: list[dict[str, Any]] = []
    for forecast_hour in forecast_hours:
        herbie = Herbie(init_ts.to_pydatetime(), fxx=int(forecast_hour), **_herbie_kwargs_for(model, cfg))
        for variable in variables:
            spec = _spec_for(variable)
            ds = herbie.xarray(spec.search, remove_grib=False)
            data_array = _first_data_var(ds)
            point_value = _point_value_from_dataarray(data_array, station_lat=station_lat, station_lon=station_lon)
            rows.append(
                {
                    "valid_time_utc": _extract_valid_time(ds, init_ts, forecast_hour),
                    "var_name": spec.output_name,
                    "value": _convert_value(point_value, spec.unit),
                    "model": model,
                    "init_run_utc": init_ts,
                    "forecast_hour": int(forecast_hour),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["valid_time_utc", "var_name", "value", "model", "init_run_utc", "forecast_hour"])

    frame.sort_values(["forecast_hour", "var_name"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


@click.command()
@click.option("--model", required=True, type=click.Choice(sorted(_MODEL_DEFAULTS)))
@click.option("--init-run", "init_run", required=True, help="UTC init run, e.g. 2026-05-15T18:00:00Z")
@click.option("--forecast-hours", required=True, help="Comma-separated list, e.g. 18,19,20")
@click.option("--lat", required=True, type=float)
@click.option("--lon", required=True, type=float)
@click.option("--variables", default="TMP:2 m,APCP:surface", show_default=True)
def main(model: str, init_run: str, forecast_hours: str, lat: float, lon: float, variables: str) -> None:
    hours = [int(part.strip()) for part in forecast_hours.split(",") if part.strip()]
    var_list = [part.strip() for part in variables.split(",") if part.strip()]
    frame = get_model_point_forecast(
        model=model,
        init_run_utc=init_run,
        forecast_hours=hours,
        station_lat=lat,
        station_lon=lon,
        variables=var_list,
    )
    print(frame.to_csv(index=False))


if __name__ == "__main__":
    main()
