import pandas as pd

from wt.models.predict import (
    NormalDistribution,
    QuantileDistribution,
    bucket_probabilities,
    predict_distribution,
)
from wt.models.train import TrainedModel


class ConstantEstimator:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame: pd.DataFrame):
        return [self.value for _ in range(len(frame))]


def test_quantile_distribution_bucket_probabilities_sum_to_one():
    dist = QuantileDistribution(
        point=70.0,
        quantiles={0.05: 60.0, 0.5: 70.0, 0.95: 80.0},
        sigma=4.0,
    )
    probs = bucket_probabilities(dist, [(-float('inf'), 65.0), (65.0, 75.0), (75.0, float('inf'))])
    assert round(sum(probs), 8) == 1.0
    assert probs[1] > probs[0]


def test_predict_distribution_prefers_quantile_heads():
    model = TrainedModel(
        station='KNYC',
        target='tmax',
        feature_names=['x'],
        point_model=ConstantEstimator(72.0),
        quantile_models={
            0.05: ConstantEstimator(65.0),
            0.5: ConstantEstimator(72.0),
            0.95: ConstantEstimator(79.0),
        },
        residual_sigma=3.0,
        metrics={},
        feature_importance={},
    )
    dist = predict_distribution(model, {'x': 1.0})
    assert isinstance(dist, QuantileDistribution)
    assert dist.point == 72.0


def test_normal_distribution_pmf_is_positive():
    dist = NormalDistribution(point=0.0, sigma=1.0)
    assert 0.68 < dist.pmf_bucket(-1.0, 1.0) < 0.69
