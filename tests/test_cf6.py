from datetime import UTC, datetime, date
from pathlib import Path

import pandas as pd

from wt.io.cf6 import TRACE_PRECIP_IN, fetch_cf6, parse_cf6_text


FIXTURE = Path(__file__).parent / "fixtures" / "cf6" / "sample_okx_sep2025.txt"


def test_parse_cf6_text_extracts_expected_values() -> None:
    parsed = parse_cf6_text(FIXTURE.read_text(encoding="utf-8"))

    assert list(parsed["local_date"][:2]) == [date(2025, 9, 1), date(2025, 9, 2)]
    assert parsed.loc[0, "tmax_f"] == 84.0
    assert parsed.loc[3, "precip_in"] == TRACE_PRECIP_IN
    assert pd.isna(parsed.loc[4, "tmax_f"])
    assert parsed.loc[4, "precip_in"] == 0.12
    assert parsed.loc[0, "settled_at"] == pd.Timestamp(datetime(2025, 9, 2, 9, 10, tzinfo=UTC))


def test_fetch_cf6_maps_live_schema(monkeypatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "valid": "2025-09-04",
                        "high": 79,
                        "low": 65,
                        "precip": "T",
                        "snow": 0.0,
                        "product": "202509020910-KOKX-CXUS51-CF6NYC",
                    },
                    {
                        "valid": "2025-10-01",
                        "high": 70,
                        "low": 58,
                        "precip": 0.0,
                        "snow": 0.0,
                        "product": "202510010910-KOKX-CXUS51-CF6NYC",
                    },
                ]
            }

    def fake_get(*args, **kwargs):
        return DummyResponse()

    monkeypatch.setattr("wt.io.cf6.requests.get", fake_get)
    frame = fetch_cf6("OKX", "KNYC", 2025, 9)

    assert list(frame["local_date"]) == [date(2025, 9, 4)]
    assert frame.loc[0, "precip_in"] == TRACE_PRECIP_IN
    assert frame.loc[0, "settled_at"] == pd.Timestamp(datetime(2025, 9, 2, 9, 10, tzinfo=UTC))
