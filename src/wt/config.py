"""Project configuration loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wt.utils.paths import CONFIG_DIR, DATA_DIR, MODELS_DIR


class Station(BaseModel):
    """Canonical station metadata used across ingestion and modeling."""

    kalshi_city: str
    icao: str
    ghcnd_id: str
    lat: float
    lon: float
    tz: str
    wfo: str

    @property
    def zoneinfo(self) -> ZoneInfo:
        return ZoneInfo(self.tz)


class AppSettings(BaseSettings):
    """Environment-backed runtime settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Field(default=DATA_DIR, alias="DATA_DIR")
    models_dir: Path = Field(default=MODELS_DIR, alias="MODELS_DIR")
    herbie_save_dir: Path = Field(default=DATA_DIR / "raw" / "herbie", alias="HERBIE_SAVE_DIR")
    herbie_config_path: Path = Field(
        default=DATA_DIR / "interim" / "herbie" / "config.toml",
        alias="HERBIE_CONFIG_PATH",
    )
    mplconfigdir: Path = Field(
        default=DATA_DIR / "interim" / "matplotlib",
        alias="MPLCONFIGDIR",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    nws_user_agent: str = Field(default="weather-trader (you@example.com)", alias="NWS_USER_AGENT")
    kalshi_api_key_id: str | None = Field(default=None, alias="KALSHI_API_KEY_ID")
    kalshi_private_key_path: Path | None = Field(default=None, alias="KALSHI_PRIVATE_KEY_PATH")
    polymarket_api_url: str = Field(default="https://clob.polymarket.com", alias="POLYMARKET_API_URL")
    supabase_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"),
    )
    supabase_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"),
    )
    supabase_publishable_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SUPABASE_PUBLISHABLE_KEY",
            "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY",
        ),
    )
    supabase_db_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("POSTGRES_URL", "SUPABASE_DB_URL"),
    )
    supabase_db_url_non_pooling: str | None = Field(
        default=None,
        validation_alias=AliasChoices("POSTGRES_URL_NON_POOLING", "SUPABASE_DB_URL_NON_POOLING"),
    )
    supabase_host: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_HOST"))
    supabase_database: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DATABASE"))
    supabase_user: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_USER"))
    supabase_password: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_PASSWORD"))
    mongodb_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MONGODB_URI", "MONGODB_ATLAS_URI"),
    )
    motherduck_token: str | None = Field(default=None, validation_alias=AliasChoices("MOTHERDUCK_TOKEN"))
    motherduck_readonly_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MOTHERDUCK_READONLY_TOKEN"),
    )
    motherduck_database: str = Field(default="wt", alias="MOTHERDUCK_DATABASE")


class StationConfig(BaseModel):
    stations: list[Station]


class DeploymentConfig(BaseModel):
    frontend: dict[str, Any]
    schedulers: dict[str, Any]


class StorageTargets(BaseModel):
    primary: str
    mirror_local_parquet: bool | None = None
    alternative: str | None = None


class RetentionConfig(BaseModel):
    local_raw_days: int = 30
    local_interim_days: int = 30
    local_feature_days: int = 180
    local_prediction_days: int = 365
    local_signal_days: int = 365
    local_log_days: int = 30
    keep_model_versions: int = 4
    keep_labels_forever: bool = True
    keep_market_snapshots_days: int = 30


class MotherDuckConfig(BaseModel):
    enabled: bool = False
    database: str = "wt"
    schema_name: str = Field(default="main", alias="schema")


class StorageConfig(BaseModel):
    deployment: DeploymentConfig
    storage: dict[str, StorageTargets]
    retention: RetentionConfig
    motherduck: MotherDuckConfig


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    return AppSettings()


@lru_cache(maxsize=1)
def load_stations(config_path: Path | None = None) -> list[Station]:
    path = config_path or CONFIG_DIR / "stations.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    parsed = StationConfig.model_validate(data)
    return parsed.stations


@lru_cache(maxsize=1)
def station_map(config_path: Path | None = None) -> dict[str, Station]:
    return {station.icao: station for station in load_stations(config_path)}


@lru_cache(maxsize=4)
def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def load_storage_config(config_path: Path | None = None) -> StorageConfig:
    path = config_path or CONFIG_DIR / "storage.yaml"
    return StorageConfig.model_validate(load_yaml(path))
