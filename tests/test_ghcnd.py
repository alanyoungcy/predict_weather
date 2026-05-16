from datetime import date

from wt.io.ghcnd import fetch_ghcnd_station


CSV_TEXT = """DATE,TMAX,TMIN,PRCP\n2025-01-01,-17,-72,191\n2025-01-02,-28,-61,8\n"""


class DummyResponse:
    text = CSV_TEXT

    def raise_for_status(self) -> None:
        return None


def test_fetch_ghcnd_station_converts_metric_tenths(monkeypatch) -> None:
    monkeypatch.setattr("wt.io.ghcnd.requests.get", lambda *args, **kwargs: DummyResponse())
    frame = fetch_ghcnd_station("USW00094728", start=date(2025, 1, 1), end=date(2025, 1, 2))

    assert list(frame["date"]) == [date(2025, 1, 1), date(2025, 1, 2)]
    assert round(frame.loc[0, "tmax_f"], 2) == 28.94
    assert round(frame.loc[0, "tmin_f"], 2) == 19.04
    assert round(frame.loc[0, "prcp_in"], 4) == 0.7520
