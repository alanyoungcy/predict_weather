"""Read-only Kalshi venue client and market ingestion."""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import pandas as pd

from wt.config import Station, load_settings, load_yaml
from wt.markets.buckets import extract_buckets_from_market
from wt.utils.paths import CONFIG_DIR


KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_TICKER_DATE_RE = re.compile(
    r"-(?P<yy>\d{2})(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?P<dd>\d{1,2})",
    re.IGNORECASE,
)
_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True, slots=True)
class Market:
    venue: str
    ticker: str
    title: str | None
    subtitle: str | None
    status: str | None
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OrderBook:
    venue: str
    ticker: str
    yes: list[list[float]]
    no: list[list[float]]
    raw: dict[str, Any]

    @property
    def best_yes_ask(self) -> float | None:
        return _best_ask(self.yes)

    @property
    def best_no_ask(self) -> float | None:
        return _best_ask(self.no)


def _best_ask(levels: list[list[float]]) -> float | None:
    if not levels:
        return None
    prices = [float(level[0]) for level in levels if level]
    if not prices:
        return None
    best = min(prices)
    return best / 100.0 if best > 1.0 else best


def _load_private_key(path: Path):
    try:
        from cryptography.hazmat.primitives import serialization
    except Exception as exc:
        raise RuntimeError("cryptography is required for authenticated Kalshi requests") from exc
    with path.open("rb") as handle:
        return serialization.load_pem_private_key(handle.read(), password=None)


class KalshiClient:
    """Small read-only REST client.

    Public endpoints are attempted without credentials. If ``api_key_id`` and a
    PEM private key are configured, requests are signed with Kalshi's ED25519
    header scheme.
    """

    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | Path | None = None,
        *,
        base_url: str = KALSHI_API_BASE,
        timeout: float = 20.0,
    ) -> None:
        settings = load_settings()
        self.api_key_id = api_key_id or settings.kalshi_api_key_id
        key_path = private_key_path or settings.kalshi_private_key_path
        self.private_key_path = Path(key_path) if key_path else None
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._private_key = None
        if self.api_key_id and self.private_key_path:
            self._private_key = _load_private_key(self.private_key_path)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _headers(self, method: str, path: str) -> dict[str, str]:
        if not self.api_key_id or self._private_key is None:
            return {}
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(message)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
        }

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = f"?{urlencode(params, doseq=True)}" if params else ""
        response = self.client.get(
            path,
            params=params,
            headers=self._headers("GET", f"{path}{query}"),
        )
        response.raise_for_status()
        return response.json()

    def list_markets(
        self,
        series_ticker: str | None = None,
        *,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
    ) -> list[Market]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        payload = self._get("/markets", params)
        return [_market_from_payload(item) for item in payload.get("markets", [])]

    def get_market(self, ticker: str) -> Market:
        payload = self._get(f"/markets/{ticker}")
        return _market_from_payload(payload.get("market", payload))

    def get_orderbook(self, ticker: str) -> OrderBook:
        payload = self._get(f"/markets/{ticker}/orderbook")
        book = payload.get("orderbook", payload)
        return OrderBook(
            venue="kalshi",
            ticker=ticker,
            yes=book.get("yes", []) or [],
            no=book.get("no", []) or [],
            raw=payload,
        )

    def fetch_relevant_markets(
        self,
        stations: list[Station],
        target_local_date: date,
        *,
        targets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch open Kalshi weather buckets configured for the station list."""

        return fetch_relevant_markets(
            self,
            stations=stations,
            target_local_date=target_local_date,
            targets=targets,
        )


def _market_from_payload(item: dict[str, Any]) -> Market:
    return Market(
        venue="kalshi",
        ticker=str(item.get("ticker", "")),
        title=item.get("title"),
        subtitle=item.get("subtitle"),
        status=item.get("status"),
        raw=item,
    )


def parse_kalshi_ticker_date(ticker: str) -> date | None:
    """Parse Kalshi weather ticker dates like ``KXHIGHNY-26MAY16``."""

    match = _TICKER_DATE_RE.search(ticker)
    if not match:
        return None
    year = 2000 + int(match.group("yy"))
    month = _MONTHS[match.group("mon").upper()]
    day = int(match.group("dd"))
    return date(year, month, day)


def _configured_series(target: str, city: str) -> str | None:
    cfg = load_yaml(CONFIG_DIR / "targets.yaml")
    return (
        cfg.get("targets", {})
        .get(target, {})
        .get("venues", {})
        .get("kalshi", {})
        .get("series_by_city", {})
        .get(city)
    )


def _market_bucket_rows(
    client: KalshiClient,
    market: Market,
    *,
    station: Station,
    target: str,
    target_local_date: date,
) -> list[dict[str, Any]]:
    payload = {
        **market.raw,
        "ticker": market.ticker,
        "title": market.title,
        "subtitle": market.subtitle,
    }
    buckets = extract_buckets_from_market(
        payload,
        venue="kalshi",
        station=station.icao,
        target=target,
    )
    if not buckets:
        return []
    try:
        orderbook = client.get_orderbook(market.ticker)
        yes_ask = orderbook.best_yes_ask
        no_ask = orderbook.best_no_ask
    except Exception:
        yes_ask = payload.get("yes_ask") or payload.get("yes_ask_dollars") or payload.get("yes_bid")
        no_ask = payload.get("no_ask") or payload.get("no_ask_dollars") or payload.get("no_bid")

    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        rows.append(
            {
                "venue": "kalshi",
                "ticker": bucket.ticker or market.ticker,
                "station": station.icao,
                "target": target,
                "target_local_date": target_local_date,
                "bucket_lo": bucket.lo,
                "bucket_hi": bucket.hi,
                "yes_ask": bucket.yes_ask if bucket.yes_ask is not None else yes_ask,
                "no_ask": bucket.no_ask if bucket.no_ask is not None else no_ask,
                "title": bucket.title or market.title,
            }
        )
    return rows


def fetch_relevant_markets(
    client: KalshiClient,
    *,
    stations: list[Station],
    target_local_date: date,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch configured Kalshi market rows with parsed bucket bounds and asks."""

    target_list = targets or ["tmax", "tmin", "apcp24"]
    rows: list[dict[str, Any]] = []
    for station in stations:
        for target in target_list:
            series = _configured_series(target, station.kalshi_city)
            if not series:
                continue
            for market in client.list_markets(series_ticker=series, status="open"):
                market_date = parse_kalshi_ticker_date(market.ticker)
                if market_date is not None and market_date != target_local_date:
                    continue
                rows.extend(
                    _market_bucket_rows(
                        client,
                        market,
                        station=station,
                        target=target,
                        target_local_date=target_local_date,
                    )
                )
    return pd.DataFrame(rows)
