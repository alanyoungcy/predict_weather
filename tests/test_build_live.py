from datetime import date

import pandas as pd

from wt.config import Station
from wt.features.build_live import build_live_features


def test_build_live_features_with_mock_fetcher():
    station = Station(
        kalshi_city='NYC',
        icao='KNYC',
        ghcnd_id='USW00094728',
        lat=40.7789,
        lon=-73.9692,
        tz='America/New_York',
        wfo='OKX',
    )

    def fake_fetcher(model, init_run, hours, lat, lon, variables):
        rows = []
        for hour in hours[:2]:
            rows.append(
                {
                    'valid_time_utc': pd.Timestamp(init_run) + pd.Timedelta(hours=hour),
                    'var_name': 't2m_f',
                    'value': 70.0,
                    'model': model,
                    'init_run_utc': pd.Timestamp(init_run),
                    'forecast_hour': hour,
                }
            )
            rows.append(
                {
                    'valid_time_utc': pd.Timestamp(init_run) + pd.Timedelta(hours=hour),
                    'var_name': 'apcp_in',
                    'value': 0.01,
                    'model': model,
                    'init_run_utc': pd.Timestamp(init_run),
                    'forecast_hour': hour,
                }
            )
        return pd.DataFrame(rows)

    features = build_live_features(
        [station],
        date(2026, 5, 18),
        forecast_fetcher=fake_fetcher,
        use_fallbacks=False,
    )
    assert len(features) == 1
    assert features.loc[0, 'station'] == 'KNYC'
    assert 'hrrr_t2m_max_f' in features.columns
    assert 'model_spread_tmax_f' in features.columns
