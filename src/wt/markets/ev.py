"""Expected-value and sizing calculations for prediction-market contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EVResult:
    ev_yes: float
    ev_no: float
    edge_bps: int
    side: str
    kelly_fraction: float


def _clean_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clean_price(value: float | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed > 1.0:
        parsed /= 100.0
    return max(0.0, min(1.0, parsed))


def expected_value(
    model_prob: float,
    yes_price: float,
    no_price: float | None = None,
    *,
    fee_rate: float = 0.0,
) -> tuple[float, float]:
    """Return YES and NO expected value per $1 payout contract.

    Prices are dollars in [0, 1]. When a NO ask is unavailable, the function
    uses the complement of YES price as an approximation.
    """

    p_yes = _clean_probability(model_prob)
    p_no = 1.0 - p_yes
    yes_cost = _clean_price(yes_price)
    no_cost = _clean_price(no_price)
    if yes_cost is None:
        raise ValueError("yes_price is required")
    if no_cost is None:
        no_cost = 1.0 - yes_cost

    fee_rate = max(0.0, float(fee_rate))
    ev_yes = p_yes * (1.0 - fee_rate * max(0.0, 1.0 - yes_cost)) - yes_cost
    ev_no = p_no * (1.0 - fee_rate * max(0.0, 1.0 - no_cost)) - no_cost
    return ev_yes, ev_no


def half_kelly_fraction(probability: float, price: float, *, cap: float = 0.02) -> float:
    """Return half-Kelly stake fraction for a binary $1 payout contract."""

    p = _clean_probability(probability)
    cost = _clean_price(price)
    if cost is None or cost <= 0.0 or cost >= 1.0:
        return 0.0
    b = (1.0 - cost) / cost
    full_kelly = (b * p - (1.0 - p)) / b
    return max(0.0, min(float(cap), 0.5 * full_kelly))


def evaluate_contract(
    model_prob: float,
    yes_price: float,
    no_price: float | None = None,
    *,
    fee_rate: float = 0.0,
    max_position_pct: float = 0.02,
) -> EVResult:
    """Compute EV, preferred side, edge in bps, and capped half-Kelly size."""

    ev_yes, ev_no = expected_value(model_prob, yes_price, no_price, fee_rate=fee_rate)
    side = "YES" if ev_yes >= ev_no else "NO"
    edge = ev_yes if side == "YES" else ev_no
    side_prob = model_prob if side == "YES" else 1.0 - model_prob
    side_price = (
        yes_price
        if side == "YES"
        else (no_price if no_price is not None else 1.0 - yes_price)
    )
    return EVResult(
        ev_yes=float(ev_yes),
        ev_no=float(ev_no),
        edge_bps=int(round(edge * 10_000)),
        side=side,
        kelly_fraction=half_kelly_fraction(side_prob, float(side_price), cap=max_position_pct),
    )
