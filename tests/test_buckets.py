from math import inf

from wt.markets.buckets import extract_buckets_from_market, parse_bucket_range
from wt.models.predict import NormalDistribution
from wt.markets.buckets import align_predictions_to_buckets, bucket_probabilities


def test_parse_open_ended_bucket_text():
    assert parse_bucket_range('Above 95 degrees') == (95.0, inf)
    assert parse_bucket_range('Below 30°') == (-inf, 30.0)


def test_extract_buckets_from_market_outcomes():
    market = {
        'ticker': 'KXHIGHNY-26MAY18',
        'outcomes': [
            {'ticker': 'A', 'title': 'Below 70'},
            {'ticker': 'B', 'title': '70 to 72'},
            {'ticker': 'C', 'title': 'Above 72'},
        ],
    }
    buckets = extract_buckets_from_market(market, station='KNYC', target='tmax')
    assert [bucket.ticker for bucket in buckets] == ['A', 'B', 'C']
    assert buckets[1].lo == 70.0
    assert buckets[1].hi == 72.0


def test_bucket_probabilities_sum_to_one():
    buckets = extract_buckets_from_market(
        {
            'buckets': [
                {'ticker': 'A', 'lo': -inf, 'hi': 70},
                {'ticker': 'B', 'lo': 70, 'hi': inf},
            ]
        }
    )
    probs = bucket_probabilities(NormalDistribution(point=70, sigma=2), buckets)
    assert round(sum(probs), 8) == 1.0


def test_align_predictions_to_buckets_builds_model_prob_table():
    import pandas as pd

    predictions = pd.DataFrame(
        [
            {
                'station': 'KNYC',
                'target': 'tmax',
                'target_local_date': '2026-05-18',
                'point_pred': 70.0,
                'q05': 65.0,
                'q25': 68.0,
                'q50': 70.0,
                'q75': 72.0,
                'q95': 75.0,
                'sigma': 3.0,
                'dist_family': 'quantile',
            }
        ]
    )
    buckets = pd.DataFrame(
        [
            {
                'venue': 'kalshi',
                'ticker': 'A',
                'station': 'KNYC',
                'target': 'tmax',
                'target_local_date': '2026-05-18',
                'bucket_lo': -inf,
                'bucket_hi': 70.0,
            },
            {
                'venue': 'kalshi',
                'ticker': 'B',
                'station': 'KNYC',
                'target': 'tmax',
                'target_local_date': '2026-05-18',
                'bucket_lo': 70.0,
                'bucket_hi': inf,
            },
        ]
    )
    aligned = align_predictions_to_buckets(predictions, buckets)
    assert len(aligned) == 2
    assert round(aligned['model_prob'].sum(), 8) == 1.0
