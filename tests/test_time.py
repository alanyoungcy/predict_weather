from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from wt.utils.time import kalshi_day_window_utc


def test_kalshi_day_window_uses_standard_offset_during_dst() -> None:
    start_utc, end_utc = kalshi_day_window_utc(date(2026, 7, 15), ZoneInfo("America/New_York"))
    assert start_utc == datetime(2026, 7, 15, 5, 0, tzinfo=UTC)
    assert end_utc == datetime(2026, 7, 16, 5, 0, tzinfo=UTC)
