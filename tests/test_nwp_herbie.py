from datetime import UTC, datetime

import pandas as pd
import xarray as xr

from wt.io import nwp_herbie


class FakeHerbie:
    seen_dates = []
    return_dataset_list = False

    def __init__(self, date, *, model, fxx, product, save_dir, verbose):
        self.seen_dates.append(date)
        self.date = date
        self.model = model
        self.fxx = fxx
        self.product = product

    def xarray(self, search, remove_grib=False):
        if search == 'TMP:2 m':
            dataset = xr.Dataset(
                data_vars={
                    't2m': xr.DataArray(
                        [[273.15, 274.15], [275.15, 276.15]],
                        dims=('y', 'x'),
                        coords={
                            'latitude': (('y', 'x'), [[40.0, 40.0], [41.0, 41.0]]),
                            'longitude': (('y', 'x'), [[-74.0, -73.0], [-74.0, -73.0]]),
                            'valid_time': pd.Timestamp('2026-05-15T18:00:00Z'),
                        },
                    )
                }
            )
            return [xr.Dataset(), dataset] if self.return_dataset_list else dataset
        if search == 'APCP:surface':
            dataset = xr.Dataset(
                data_vars={
                    'tp': xr.DataArray(
                        [10.0, 20.0],
                        dims=('latitude',),
                        coords={
                            'latitude': [40.0, 41.0],
                            'longitude': ('latitude', [-74.0, -73.0]),
                            'valid_time': pd.Timestamp('2026-05-15T19:00:00Z'),
                        },
                    )
                }
            )
            return [dataset] if self.return_dataset_list else dataset
        raise AssertionError(search)


def test_get_model_point_forecast_converts_units_and_shapes(monkeypatch, tmp_path):
    FakeHerbie.seen_dates = []
    FakeHerbie.return_dataset_list = False
    monkeypatch.setattr(nwp_herbie, '_load_herbie_class', lambda: FakeHerbie)
    settings = type('Settings', (), {
        'herbie_save_dir': tmp_path / 'herbie',
        'herbie_config_path': tmp_path / 'herbie' / 'config.toml',
    })()

    frame = nwp_herbie.get_model_point_forecast(
        model='hrrr',
        init_run_utc=datetime(2026, 5, 15, 18, tzinfo=UTC),
        forecast_hours=[1],
        station_lat=40.9,
        station_lon=-73.1,
        variables=['TMP:2 m', 'APCP:surface'],
        settings=settings,
    )

    assert list(frame['var_name']) == ['apcp_in', 't2m_f']
    assert frame.loc[frame['var_name'] == 't2m_f', 'value'].iloc[0] == 37.4
    assert round(frame.loc[frame['var_name'] == 'apcp_in', 'value'].iloc[0], 6) == 0.787402
    assert set(frame.columns) == {'valid_time_utc', 'var_name', 'value', 'model', 'init_run_utc', 'forecast_hour'}
    assert FakeHerbie.seen_dates[0].tzinfo is None
    assert frame['init_run_utc'].iloc[0].tzinfo is not None


def test_get_model_point_forecast_handles_herbie_dataset_lists(monkeypatch, tmp_path):
    FakeHerbie.seen_dates = []
    FakeHerbie.return_dataset_list = True
    monkeypatch.setattr(nwp_herbie, '_load_herbie_class', lambda: FakeHerbie)
    settings = type('Settings', (), {
        'herbie_save_dir': tmp_path / 'herbie',
        'herbie_config_path': tmp_path / 'herbie' / 'config.toml',
    })()

    frame = nwp_herbie.get_model_point_forecast(
        model='gfs',
        init_run_utc=datetime(2026, 5, 15, 18, tzinfo=UTC),
        forecast_hours=[1],
        station_lat=40.9,
        station_lon=-73.1,
        variables=['TMP:2 m', 'APCP:surface'],
        settings=settings,
    )

    assert list(frame['var_name']) == ['apcp_in', 't2m_f']
    assert frame['valid_time_utc'].notna().all()
