"""Read-only Polymarket venue client and direct-weather market discovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from typing import Any

import httpx
import pandas as pd

from wt.config import Station, load_settings, load_yaml
from wt.markets.buckets import extract_buckets_from_market
from wt.utils.paths import CONFIG_DIR


POLYMARKET_CLOB_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass(frozen=True, slots=True)
class PolymarketMarket:
    venue: str
    condition_id: str
    question: str
    slug: str | None
    active: bool
    closed: bool
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PolymarketOrderBook:
    venue: str
    token_id: str
    bids: list[dict[str, Any]]
    asks: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def best_ask(self) -> float | None:
        prices = [float(level["price"]) for level in self.asks if level.get("price") is not None]
        return min(prices) if prices else None


class PolymarketClient:
    """Read-only Polymarket Gamma + CLOB client.

    v1 only discovers direct daily-weather analogs when they exist. It does not
    place or cancel orders.
    """

    def __init__(
        self,
        *,
        clob_url: str | None = None,
        gamma_url: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        settings = load_settings()
        self.clob_url = (
            clob_url or settings.polymarket_api_url or POLYMARKET_CLOB_BASE
        ).rstrip("/")
        self.gamma_url = (
            gamma_url or settings.polymarket_gamma_api_url or POLYMARKET_GAMMA_BASE
        ).rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "PolymarketClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def search_markets(
        self,
        query: str = "weather",
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
    ) -> list[PolymarketMarket]:
        response = self.client.get(
            f"{self.gamma_url}/markets",
            params={
                "search": query,
                "active": str(active).lower(),
                "closed": str(closed).lower(),
                "limit": limit,
            },
        )
        response.raise_for_status()
        payload = response.json()
        items = payload if isinstance(payload, list) else payload.get("markets", [])
        return [_market_from_payload(item) for item in items]

    def find_daily_weather_markets(
        self,
        *,
        city: str | None = None,
        target: str | None = None,
    ) -> list[PolymarketMarket]:
        terms = ["weather", "temperature", "rain"]
        if city:
            terms.append(city)
        if target:
            terms.append(target)
        markets = self.search_markets(" ".join(terms))
        return [market for market in markets if _looks_like_daily_weather(market.question)]

    def get_orderbook(self, token_id: str) -> PolymarketOrderBook:
        response = self.client.get(f"{self.clob_url}/book", params={"token_id": token_id})
        response.raise_for_status()
        payload = response.json()
        return PolymarketOrderBook(
            venue="polymarket",
            token_id=token_id,
            bids=payload.get("bids", []) or [],
            asks=payload.get("asks", []) or [],
            raw=payload,
        )

    def fetch_relevant_markets(
        self,
        stations: list[Station],
        target_local_date: date,
        *,
        targets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch configured direct daily-weather analogs from Polymarket."""

        return fetch_relevant_markets(
            self,
            stations=stations,
            target_local_date=target_local_date,
            targets=targets,
        )


def _market_from_payload(item: dict[str, Any]) -> PolymarketMarket:
    return PolymarketMarket(
        venue="polymarket",
        condition_id=str(item.get("conditionId", item.get("condition_id", item.get("id", "")))),
        question=str(item.get("question", item.get("title", ""))),
        slug=item.get("slug"),
        active=bool(item.get("active", True)),
        closed=bool(item.get("closed", False)),
        raw=item,
    )


def _looks_like_daily_weather(question: str) -> bool:
    text = question.lower()
    weather_term = any(
        term in text for term in ("temperature", "rain", "precip", "weather", "high", "low")
    )
    daily_term = any(term in text for term in ("today", "tomorrow", "daily", "on "))
    excluded = any(
        term in text
        for term in ("hurricane", "tornado", "snowfall", "monthly", "annual", "yearly")
    )
    return weather_term and daily_term and not excluded


def _configured_slug(target: str, city: str) -> str | None:
    cfg = load_yaml(CONFIG_DIR / "targets.yaml")
    return (
        cfg.get("targets", {})
        .get(target, {})
        .get("venues", {})
        .get("polymarket", {})
        .get("slug_by_city", {})
        .get(city)
    )


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _token_payloads(market: PolymarketMarket) -> list[dict[str, Any]]:
    raw = market.raw
    tokens = _json_list(raw.get("tokens"))
    if tokens and all(isinstance(item, dict) for item in tokens):
        return [dict(item) for item in tokens]

    outcomes = _json_list(raw.get("outcomes"))
    token_ids = _json_list(raw.get("clobTokenIds")) or _json_list(raw.get("clob_token_ids"))
    prices = _json_list(raw.get("outcomePrices")) or _json_list(raw.get("outcome_prices"))
    payloads: list[dict[str, Any]] = []
    for idx, outcome in enumerate(outcomes):
        payloads.append(
            {
                "title": str(outcome),
                "token_id": str(token_ids[idx]) if idx < len(token_ids) else "",
                "yes_ask": float(prices[idx]) if idx < len(prices) else None,
            }
        )
    return payloads


def _market_rows(
    client: PolymarketClient,
    market: PolymarketMarket,
    *,
    station: Station,
    target: str,
    target_local_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token in _token_payloads(market):
        token["question"] = f"{market.question} {token.get('title', '')}"
        buckets = extract_buckets_from_market(
            token,
            venue="polymarket",
            station=station.icao,
            target=target,
        )
        if not buckets:
            continue
        token_id = str(token.get("token_id") or token.get("tokenId") or token.get("id") or "")
        yes_ask = token.get("yes_ask")
        if token_id:
            try:
                yes_ask = client.get_orderbook(token_id).best_ask or yes_ask
            except Exception:
                pass
        for bucket in buckets:
            rows.append(
                {
                    "venue": "polymarket",
                    "ticker": bucket.ticker or token_id or market.condition_id,
                    "station": station.icao,
                    "target": target,
                    "target_local_date": target_local_date,
                    "bucket_lo": bucket.lo,
                    "bucket_hi": bucket.hi,
                    "yes_ask": bucket.yes_ask if bucket.yes_ask is not None else yes_ask,
                    "no_ask": bucket.no_ask,
                    "title": bucket.title or market.question,
                }
            )
    return rows


def fetch_relevant_markets(
    client: PolymarketClient,
    *,
    stations: list[Station],
    target_local_date: date,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch configured Polymarket direct-weather rows.

    Polymarket has no stable Kalshi-style weather series. v1 only uses markets
    explicitly mapped in ``config/targets.yaml``; empty mappings produce no rows.
    """

    target_list = targets or ["tmax", "tmin", "apcp24"]
    rows: list[dict[str, Any]] = []
    for station in stations:
        for target in target_list:
            slug = _configured_slug(target, station.kalshi_city)
            markets = (
                [market for market in client.search_markets(slug, limit=20) if market.slug == slug]
                if slug
                else []
            )
            for market in markets:
                if not _looks_like_daily_weather(market.question):
                    continue
                rows.extend(
                    _market_rows(
                        client,
                        market,
                        station=station,
                        target=target,
                        target_local_date=target_local_date,
                    )
                )
    return pd.DataFrame(rows)
