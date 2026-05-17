from datetime import date

from wt.config import Station
from wt.io.kalshi import Market, OrderBook, fetch_relevant_markets, parse_kalshi_ticker_date
from wt.io.polymarket import PolymarketMarket, fetch_relevant_markets as fetch_poly_markets


def _station() -> Station:
    return Station(
        kalshi_city="NYC",
        icao="KNYC",
        ghcnd_id="USW00094728",
        lat=40.7789,
        lon=-73.9692,
        tz="America/New_York",
        wfo="OKX",
    )


def test_parse_kalshi_ticker_date():
    assert parse_kalshi_ticker_date("KXHIGHNY-26MAY18-T72") == date(2026, 5, 18)


def test_kalshi_fetch_relevant_markets_with_fake_client():
    class FakeClient:
        def list_markets(self, series_ticker, status):
            assert series_ticker == "KXHIGHNY"
            assert status == "open"
            return [
                Market(
                    venue="kalshi",
                    ticker="KXHIGHNY-26MAY18",
                    title="NYC high 70 to 72",
                    subtitle=None,
                    status="open",
                    raw={},
                )
            ]

        def get_orderbook(self, ticker):
            return OrderBook(venue="kalshi", ticker=ticker, yes=[[55, 10]], no=[[40, 10]], raw={})

    rows = fetch_relevant_markets(
        FakeClient(),
        stations=[_station()],
        target_local_date=date(2026, 5, 18),
        targets=["tmax"],
    )
    assert len(rows) == 1
    assert rows.loc[0, "yes_ask"] == 0.55
    assert rows.loc[0, "bucket_lo"] == 70.0


def test_polymarket_fetch_relevant_markets_empty_without_configured_slug():
    class FakeClient:
        def search_markets(self, query, limit):
            raise AssertionError("should not search when no slug is configured")

    rows = fetch_poly_markets(
        FakeClient(),
        stations=[_station()],
        target_local_date=date(2026, 5, 18),
        targets=["tmax"],
    )
    assert rows.empty


def test_polymarket_fetch_relevant_markets_with_configured_slug(monkeypatch):
    def fake_load_yaml(_path):
        return {
            "targets": {
                "tmax": {
                    "venues": {
                        "polymarket": {"slug_by_city": {"NYC": "nyc-high-test"}}
                    }
                }
            }
        }

    import wt.io.polymarket as polymarket_module

    monkeypatch.setattr(polymarket_module, "load_yaml", fake_load_yaml)

    class FakeClient:
        def search_markets(self, query, limit):
            assert query == "nyc-high-test"
            return [
                PolymarketMarket(
                    venue="polymarket",
                    condition_id="c1",
                    question="Will NYC daily high temperature be 70 to 72 on May 18?",
                    slug="nyc-high-test",
                    active=True,
                    closed=False,
                    raw={"outcomes": '["70 to 72"]', "clobTokenIds": '["tok1"]'},
                )
            ]

        def get_orderbook(self, token_id):
            class Book:
                best_ask = 0.44

            return Book()

    rows = fetch_poly_markets(
        FakeClient(),
        stations=[_station()],
        target_local_date=date(2026, 5, 18),
        targets=["tmax"],
    )
    assert len(rows) == 1
    assert rows.loc[0, "venue"] == "polymarket"
    assert rows.loc[0, "yes_ask"] == 0.44
