from datetime import date

import pandas as pd

from wt.markets.signals import generate_signals


def test_generate_signals_filters_and_ranks():
    predictions = pd.DataFrame(
        [
            {
                'venue': 'kalshi',
                'ticker': 'A',
                'station': 'KNYC',
                'target': 'tmax',
                'target_local_date': date(2026, 5, 18),
                'bucket_lo': 70.0,
                'bucket_hi': 72.0,
                'model_prob': 0.7,
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                'venue': 'kalshi',
                'ticker': 'A',
                'station': 'KNYC',
                'target': 'tmax',
                'target_local_date': date(2026, 5, 18),
                'bucket_lo': 70.0,
                'bucket_hi': 72.0,
                'yes_ask': 0.55,
                'no_ask': 0.45,
            }
        ]
    )
    signals = generate_signals(predictions, markets, min_edge_bps=300)
    assert len(signals) == 1
    assert signals.loc[0, 'side'] == 'YES'
    assert signals.loc[0, 'edge_bps'] == 1500
