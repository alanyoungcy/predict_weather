from datetime import date

import pandas as pd

from wt.config import Station
from wt.features.build_train import build_training_features
from wt.features.transforms import add_model_spread, build_climatology


def _station() -> Station:
    return Station(
        kalshi_city='NYC',
        icao='KNYC',
        ghcnd_id='USW00094728',
        lat=40.7789,
        lon=-73.9692,
        tz='America/New_York',
        wfo='OKX',
    )


def test_add_model_spread_computes_std():
    frame = pd.DataFrame([{'hrrr_t2m_max_f': 70.0, 'gfs_t2m_max_f': 74.0, 'hrrr_apcp_24h_in': 0.1, 'gfs_apcp_24h_in': 0.3}])
    out = add_model_spread(frame)
    assert round(out.loc[0, 'model_spread_tmax_f'], 4) == 2.0
    assert round(out.loc[0, 'model_spread_apcp_in'], 4) == 0.1


def test_build_training_features_uses_only_prior_day_labels():
    labels = pd.DataFrame(
        [
            {'station': 'KNYC', 'local_date': date(2026, 1, 1), 'tmax_f': 40.0, 'tmin_f': 30.0, 'precip_in': 0.1},
            {'station': 'KNYC', 'local_date': date(2026, 1, 2), 'tmax_f': 41.0, 'tmin_f': 31.0, 'precip_in': 0.2},
            {'station': 'KNYC', 'local_date': date(2026, 1, 3), 'tmax_f': 99.0, 'tmin_f': 88.0, 'precip_in': 7.7},
        ]
    )
    climo = build_climatology(labels)

    def fake_fetcher(model, init_run, hours, lat, lon, variables):
        rows = []
        for hour in hours:
            rows.append({'valid_time_utc': pd.Timestamp(init_run) + pd.Timedelta(hours=hour), 'var_name': 't2m_f', 'value': 50.0, 'model': model, 'init_run_utc': pd.Timestamp(init_run), 'forecast_hour': hour})
            rows.append({'valid_time_utc': pd.Timestamp(init_run) + pd.Timedelta(hours=hour), 'var_name': 'apcp_in', 'value': 0.05, 'model': model, 'init_run_utc': pd.Timestamp(init_run), 'forecast_hour': hour})
        return pd.DataFrame(rows)

    features = build_training_features(
        station=_station(),
        start_date=date(2026, 1, 3),
        end_date=date(2026, 1, 3),
        label_df=labels,
        climo_df=climo,
        forecast_fetcher=fake_fetcher,
        models=['hrrr', 'gfs'],
    )

    assert features.loc[0, 'obs_tmax_lag1_f'] == 41.0
    assert features.loc[0, 'obs_tmin_lag1_f'] == 31.0
    assert features.loc[0, 'obs_precip_lag1_in'] == 0.2
    assert features.loc[0, 'obs_tmax_lag1_f'] != 99.0
    assert 'hrrr_t2m_max_f' in features.columns
    assert 'gfs_apcp_24h_in' in features.columns


def test_build_training_features_skips_models_before_historical_start():
    labels = pd.DataFrame(
        [
            {'station': 'KNYC', 'local_date': date(2018, 4, 30), 'tmax_f': 60.0, 'tmin_f': 45.0, 'precip_in': 0.0},
            {'station': 'KNYC', 'local_date': date(2018, 5, 1), 'tmax_f': 61.0, 'tmin_f': 46.0, 'precip_in': 0.0},
        ]
    )
    calls = []

    def fake_fetcher(model, init_run, hours, lat, lon, variables):
        calls.append(model)
        return pd.DataFrame(
            [
                {
                    'valid_time_utc': pd.Timestamp(init_run),
                    'var_name': 't2m_f',
                    'value': 50.0,
                    'model': model,
                    'init_run_utc': pd.Timestamp(init_run),
                    'forecast_hour': 1,
                }
            ]
        )

    features = build_training_features(
        station=_station(),
        start_date=date(2018, 5, 1),
        end_date=date(2018, 5, 1),
        label_df=labels,
        forecast_fetcher=fake_fetcher,
        models=['hrrr', 'gfs'],
    )

    assert calls == ['hrrr']
    assert 'hrrr_t2m_max_f' in features.columns
    assert 'gfs_t2m_max_f' not in features.columns
