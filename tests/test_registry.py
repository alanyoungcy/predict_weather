from datetime import UTC, datetime

import xgboost as xgb

from wt.models.registry import load_trained_model, model_version_tag, save_trained_model
from wt.models.train import TrainedModel



def _dummy_trained_model() -> TrainedModel:
    model = xgb.XGBRegressor(n_estimators=1, objective='reg:squarederror')
    return TrainedModel(
        station='KNYC',
        target='tmax',
        feature_names=['x1'],
        point_model=model,
        quantile_models={0.5: model},
        residual_sigma=1.2,
        metrics={'val_mae': 1.0},
        feature_importance={'x1': 1.0},
    )


def test_model_version_tag_includes_date_and_sha():
    tag = model_version_tag(datetime(2026, 5, 17, tzinfo=UTC))
    assert tag.startswith('20260517-')



def test_save_and_load_trained_model_roundtrip(tmp_path):
    trained = _dummy_trained_model()
    path = save_trained_model(trained, tmp_path)
    restored = load_trained_model(path)
    assert restored.station == 'KNYC'
    assert restored.target == 'tmax'
    assert restored.metrics['val_mae'] == 1.0
