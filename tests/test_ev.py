from wt.markets.ev import evaluate_contract, expected_value, half_kelly_fraction


def test_expected_value_uses_actual_no_ask_when_present():
    ev_yes, ev_no = expected_value(0.7, yes_price=0.55, no_price=0.35)
    assert round(ev_yes, 4) == 0.15
    assert round(ev_no, 4) == -0.05


def test_evaluate_contract_ranks_yes_edge():
    result = evaluate_contract(0.7, yes_price=0.55, no_price=0.45)
    assert result.side == 'YES'
    assert result.edge_bps == 1500
    assert result.kelly_fraction > 0


def test_half_kelly_is_capped():
    assert half_kelly_fraction(0.99, 0.01, cap=0.02) == 0.02
