"""Model training for station-specific weather targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
TARGET_MAP = {
    'tmax': 'tmax_f',
    'tmin': 'tmin_f',
    'apcp24': 'precip_in',
}
DEFAULT_BASE_PARAMS: dict[str, Any] = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.03,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'eval_metric': 'mae',
    'tree_method': 'hist',
}


@dataclass(slots=True)
class TrainedModel:
    station: str
    target: str
    feature_names: list[str]
    point_model: xgb.XGBRegressor
    quantile_models: dict[float, xgb.XGBRegressor]
    residual_sigma: float
    metrics: dict[str, float]
    feature_importance: dict[str, float]



def _target_column(target: str) -> str:
    try:
        return TARGET_MAP[target]
    except KeyError as exc:
        raise ValueError(f'Unsupported target {target!r}') from exc



def _prepare_training_frame(features: pd.DataFrame, labels: pd.DataFrame, station: str, target: str) -> pd.DataFrame:
    target_col = _target_column(target)
    left = features.copy()
    right = labels.copy()
    left['target_local_date'] = pd.to_datetime(left['target_local_date']).dt.date
    right['local_date'] = pd.to_datetime(right['local_date']).dt.date
    merged = left.merge(
        right[right['station'] == station][['station', 'local_date', target_col]],
        left_on=['station', 'target_local_date'],
        right_on=['station', 'local_date'],
        how='inner',
    )
    merged.rename(columns={target_col: 'target_value'}, inplace=True)
    merged.drop(columns=['local_date'], inplace=True)
    merged = merged[(merged['station'] == station) & merged['target_value'].notna()].copy()
    merged.sort_values('target_local_date', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged



def _date_mask(dates: pd.Series, bounds: tuple[pd.Timestamp | str, pd.Timestamp | str]) -> pd.Series:
    start, end = pd.Timestamp(bounds[0]).date(), pd.Timestamp(bounds[1]).date()
    return dates.between(start, end)



def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {'station', 'target_local_date', 'init_run_utc', 'target_value'}
    return [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]



def _fit_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    objective: str,
    extra_params: dict[str, Any] | None = None,
) -> xgb.XGBRegressor:
    params = {**DEFAULT_BASE_PARAMS, **(extra_params or {}), 'objective': objective}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model



def train_one(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    station: str,
    target: str,
    val_split: tuple[pd.Timestamp | str, pd.Timestamp | str],
    test_split: tuple[pd.Timestamp | str, pd.Timestamp | str],
) -> TrainedModel:
    """Train point + quantile regressors for one station/target pair."""

    frame = _prepare_training_frame(features, labels, station=station, target=target)
    if frame.empty:
        raise ValueError(f'No training rows available for {station} {target}')

    feature_cols = _feature_columns(frame)
    if not feature_cols:
        raise ValueError('No numeric feature columns available for training')

    dates = frame['target_local_date']
    val_mask = _date_mask(dates, val_split)
    test_mask = _date_mask(dates, test_split)
    train_mask = dates < pd.Timestamp(val_split[0]).date()

    train_frame = frame.loc[train_mask]
    val_frame = frame.loc[val_mask]
    test_frame = frame.loc[test_mask]
    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError('Train/val/test split produced an empty partition')

    X_train, y_train = train_frame[feature_cols], train_frame['target_value']
    X_val, y_val = val_frame[feature_cols], val_frame['target_value']
    X_test, y_test = test_frame[feature_cols], test_frame['target_value']

    point_model = _fit_regressor(X_train, y_train, X_val, y_val, objective='reg:squarederror')
    quantile_models: dict[float, xgb.XGBRegressor] = {}
    for tau in QUANTILES:
        quantile_models[tau] = _fit_regressor(
            X_train,
            y_train,
            X_val,
            y_val,
            objective='reg:quantileerror',
            extra_params={'quantile_alpha': tau},
        )

    val_pred = point_model.predict(X_val)
    test_pred = point_model.predict(X_test)
    residual_sigma = float(np.std(y_val - val_pred, ddof=0))
    metrics = {
        'val_mae': float(mean_absolute_error(y_val, val_pred)),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
        'test_mae': float(mean_absolute_error(y_test, test_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
        'train_rows': float(len(train_frame)),
        'val_rows': float(len(val_frame)),
        'test_rows': float(len(test_frame)),
    }
    feature_importance = {
        feature: float(score)
        for feature, score in zip(feature_cols, point_model.feature_importances_, strict=True)
    }
    return TrainedModel(
        station=station,
        target=target,
        feature_names=feature_cols,
        point_model=point_model,
        quantile_models=quantile_models,
        residual_sigma=residual_sigma,
        metrics=metrics,
        feature_importance=feature_importance,
    )
