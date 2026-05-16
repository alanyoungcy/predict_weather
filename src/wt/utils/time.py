"""Time helpers for Kalshi weather market settlement windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class KalshiWindow:
    """UTC bounds for a local-standard-time settlement day."""

    local_date: date
    timezone_name: str
    standard_offset: timedelta
    start_utc: datetime
    end_utc: datetime


_SUPPORTED_STANDARD_OFFSET_DATES = (date(2024, 1, 15), date(2024, 2, 15))


def _standard_offset(tz: ZoneInfo) -> timedelta:
    """Return the zone's standard UTC offset.

    Kalshi weather markets settle on local standard time year-round. For the US
    time zones in scope, sampling a winter date yields the non-DST offset.
    """

    for probe_date in _SUPPORTED_STANDARD_OFFSET_DATES:
        probe = datetime.combine(probe_date, time(12, 0), tzinfo=tz)
        offset = probe.utcoffset()
        dst = probe.dst()
        if offset is not None and dst == timedelta(0):
            return offset
    raise ValueError(f"Could not determine standard offset for timezone {tz.key!r}")


def kalshi_day_window_utc(local_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Return UTC bounds for the Kalshi settlement day.

    The settlement window is defined using local standard time year-round. That
    means midnight-to-midnight in the station's standard offset, even when the
    local wall clock is observing daylight saving time.
    """

    standard_tz = timezone(_standard_offset(tz), name=f"{tz.key}-standard")
    start_local_standard = datetime.combine(local_date, time.min, tzinfo=standard_tz)
    end_local_standard = start_local_standard + timedelta(days=1)
    return start_local_standard.astimezone(UTC), end_local_standard.astimezone(UTC)


def describe_window(local_date: date, tz_name: str) -> KalshiWindow:
    """Build a rich representation for logging or CLI output."""

    tz = ZoneInfo(tz_name)
    start_utc, end_utc = kalshi_day_window_utc(local_date, tz)
    return KalshiWindow(
        local_date=local_date,
        timezone_name=tz_name,
        standard_offset=_standard_offset(tz),
        start_utc=start_utc,
        end_utc=end_utc,
    )


def _format_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def main() -> None:
    """Print a known DST-sensitive example for manual verification."""

    sample_date = date(2026, 7, 15)
    window = describe_window(sample_date, "America/New_York")
    print(
        "Kalshi day window for "
        f"{window.timezone_name} on {window.local_date.isoformat()} "
        f"(standard offset { _format_timedelta(window.standard_offset) }): "
        f"{window.start_utc.isoformat()} -> {window.end_utc.isoformat()}"
    )


if __name__ == "__main__":
    main()
