"""Chaque mesure est testée sur des séries dont on connaît le résultat exact."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.risk import (
    annualized_volatility, drawdown_series, effective_number_of_lines,
    foreign_currency_weight, gaussian_var, historical_cvar, historical_var, max_drawdown,
    monte_carlo_drawdown, portfolio_returns, redistribute_missing, risk_contributions,
    stress_test, to_returns,
    weights_by_group,
)


def prices_from(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --- Rendements et poids -----------------------------------------------------

def test_to_returns():
    r = to_returns(pd.DataFrame({"X": prices_from([100, 110, 99])}))
    assert r["X"].tolist() == pytest.approx([0.10, -0.10])


def test_portfolio_returns_weighted_average():
    r = pd.DataFrame({"A": [0.10, -0.02], "B": [0.00, 0.04]})
    p = portfolio_returns(r, pd.Series({"A": 0.25, "B": 0.75}))
    assert p.tolist() == pytest.approx([0.025, 0.025])


def test_portfolio_returns_ignores_days_with_missing_asset():
    r = pd.DataFrame({"A": [0.01, 0.02], "B": [np.nan, 0.02]})
    p = portfolio_returns(r, pd.Series({"A": 0.5, "B": 0.5}))
    assert len(p) == 1                              # pas de rendement nul inventé pour B


@pytest.mark.parametrize("w", [{"A": 0.5, "B": 0.4}, {"A": 1.2, "B": -0.2}])
def test_invalid_weights_rejected(w):
    r = pd.DataFrame({"A": [0.01], "B": [0.01]})
    with pytest.raises(ValueError):
        portfolio_returns(r, pd.Series(w))


# --- Volatilité, contributions -----------------------------------------------

def test_volatility_of_two_uncorrelated_assets():
    # Deux actifs indépendants de même volatilité, 50/50 : la volatilité baisse de racine(2)
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, size=(200_000, 2)), columns=["A", "B"])
    p = portfolio_returns(r, pd.Series({"A": 0.5, "B": 0.5}))
    assert annualized_volatility(p) == pytest.approx(0.01 * np.sqrt(252) / np.sqrt(2), rel=0.01)


def test_risk_contributions_sum_to_one_and_are_equal_when_symmetric():
    rng = np.random.default_rng(2)
    r = pd.DataFrame(rng.normal(0, 0.01, size=(200_000, 3)), columns=list("ABC"))
    c = risk_contributions(r, pd.Series(1 / 3, index=list("ABC")))
    assert c.sum() == pytest.approx(1.0)
    assert c.tolist() == pytest.approx([1 / 3] * 3, abs=0.01)


def test_risky_small_line_carries_more_risk_than_its_weight():
    rng = np.random.default_rng(3)
    r = pd.DataFrame({"CALME": rng.normal(0, 0.005, 100_000),
                      "AGITE": rng.normal(0, 0.03, 100_000)})
    c = risk_contributions(r, pd.Series({"CALME": 0.9, "AGITE": 0.1}))
    # 0.9² x 0.005² = 2.0e-5 ; 0.1² x 0.03² = 0.9e-5 -> AGITE porte ~31 % du risque
    assert c["AGITE"] == pytest.approx(0.9 / 2.925, abs=0.01)


# --- VaR, CVaR ---------------------------------------------------------------

def test_historical_var_and_cvar_on_known_sample():
    # 100 rendements : -10 %, -9 %, ..., -1 %, puis 90 jours à +1 %
    r = pd.Series([-0.10 + 0.01 * i for i in range(10)] + [0.01] * 90)
    # Quantile 5 % interpolé : position 0,05 x 99 = 4,95, entre -6 % (rang 4) et -5 % (rang 5)
    assert historical_var(r, 0.95) == pytest.approx(0.0505)
    # Jours au-delà de la VaR : -10 % ... -6 % -> moyenne 8 %
    assert historical_cvar(r, 0.95) == pytest.approx(0.08)


def test_cvar_is_at_least_var():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.standard_t(df=3, size=50_000) * 0.01)
    assert historical_cvar(r) >= historical_var(r)


def test_gaussian_var_matches_theory_on_normal_data():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 500_000))
    expected = -norm.ppf(0.05) * 0.01
    assert gaussian_var(r, 0.95) == pytest.approx(expected, rel=0.01)
    assert historical_var(r, 0.95) == pytest.approx(expected, rel=0.01)


def test_gaussian_var_underestimates_fat_tails():
    # Loi de Student (queues épaisses) : à 99 %, la normale sous-estime la perte
    rng = np.random.default_rng(6)
    r = pd.Series(rng.standard_t(df=3, size=500_000) * 0.01)
    assert gaussian_var(r, 0.99) < historical_var(r, 0.99)


# --- Drawdown ----------------------------------------------------------------

def test_max_drawdown_known_path():
    # 100 -> 120 -> 60 -> 130 : pire baisse 120 -> 60 = 50 %
    r = to_returns(pd.DataFrame({"X": prices_from([100, 120, 60, 130])}))["X"]
    assert max_drawdown(r) == pytest.approx(0.50)


def test_drawdown_counts_loss_from_initial_capital():
    # Baisse dès le 1er jour, jamais de nouveau plus haut : 100 -> 80 = 20 %
    r = pd.Series([-0.10, -0.1111111111])
    assert max_drawdown(r) == pytest.approx(0.20)
    assert drawdown_series(r).iloc[0] == pytest.approx(0.10)


def test_max_drawdown_of_rising_series_is_zero():
    assert max_drawdown(pd.Series([0.01] * 50)) == 0.0


# --- Concentration -----------------------------------------------------------

def test_weights_by_group():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    sectors = pd.Series({"A": "Banks", "B": "Banks", "C": "Energy"})
    assert weights_by_group(w, sectors).to_dict() == pytest.approx({"Banks": 0.8, "Energy": 0.2})


def test_effective_number_of_lines():
    assert effective_number_of_lines(pd.Series([0.1] * 10)) == pytest.approx(10)
    assert effective_number_of_lines(pd.Series([0.91] + [0.01] * 9)) < 1.3


def test_foreign_currency_weight():
    w = pd.Series({"TTE.PA": 0.6, "MSFT": 0.3, "NESN.SW": 0.1})
    ccy = pd.Series({"TTE.PA": "EUR", "MSFT": "USD", "NESN.SW": "CHF"})
    assert foreign_currency_weight(w, ccy, "EUR") == pytest.approx(0.4)


# --- Stress test -------------------------------------------------------------

def test_stress_test_known_crash_and_partial_coverage():
    idx = pd.bdate_range("2020-02-17", "2020-03-27")
    n = len(idx)
    crash = np.r_[np.linspace(100, 60, n // 2), np.linspace(60, 70, n - n // 2)]
    prices = pd.DataFrame({
        "OLD": crash,                                  # -40 % puis rebond
        "FLAT": 100.0,
        "NEW": [np.nan] * 10 + [50.0] * (n - 10),      # coté pendant la crise seulement
    }, index=idx)
    res = stress_test(prices, pd.Series({"OLD": 0.4, "FLAT": 0.4, "NEW": 0.2}),
                      "covid", dt.date(2020, 2, 19), dt.date(2020, 3, 23))
    assert res.weight_covered == pytest.approx(0.8)    # NEW retiré
    assert res.max_drawdown > 0.15                     # moitié OLD (-40 %) moitié FLAT
    assert res.max_drawdown < 0.21


# --- Monte Carlo -------------------------------------------------------------

def test_monte_carlo_certain_loss_and_certain_gain():
    down = monte_carlo_drawdown(pd.Series([-0.001] * 500), 5, 0.20, n_simulations=200)
    up = monte_carlo_drawdown(pd.Series([0.001] * 500), 5, 0.20, n_simulations=200)
    assert down.breach_probability == 1.0              # -0,1 %/jour sur 5 ans : ~-72 %
    assert up.breach_probability == 0.0
    assert up.median_max_drawdown == 0.0


def test_monte_carlo_is_reproducible_and_seed_dependent():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0003, 0.012, 3000))
    a = monte_carlo_drawdown(r, 5, 0.20, n_simulations=500, seed=1)
    b = monte_carlo_drawdown(r, 5, 0.20, n_simulations=500, seed=1)
    c = monte_carlo_drawdown(r, 5, 0.20, n_simulations=500, seed=2)
    assert a == b
    assert a != c


def test_monte_carlo_riskier_portfolio_breaches_more():
    rng = np.random.default_rng(8)
    calm = pd.Series(rng.normal(0.0003, 0.007, 3000))
    wild = pd.Series(rng.normal(0.0003, 0.020, 3000))
    p_calm = monte_carlo_drawdown(calm, 5, 0.20, n_simulations=1000).breach_probability
    p_wild = monte_carlo_drawdown(wild, 5, 0.20, n_simulations=1000).breach_probability
    assert p_calm < p_wild


def test_monte_carlo_matches_independent_simulation():
    # Avec des blocs d'un jour, le bootstrap équivaut à des tirages indépendants :
    # on le compare à une simulation directe de rendements normaux de mêmes moments
    hist = pd.Series(np.random.default_rng(8).normal(0.0003, 0.007, 3000))
    boot = monte_carlo_drawdown(hist, 5, 0.20, n_simulations=4000, block_size=1)
    sims = np.random.default_rng(0).normal(hist.mean(), hist.std(), size=(4000, 5 * 252))
    wealth = np.cumprod(1 + sims, axis=1)
    peak = np.maximum(np.maximum.accumulate(wealth, axis=1), 1.0)
    direct = ((1 - wealth / peak).max(axis=1) > 0.20).mean()
    assert boot.breach_probability == pytest.approx(direct, abs=0.03)



# --- Titres absents pendant une crise ----------------------------------------

def test_missing_stock_weight_stays_in_its_pocket():
    w = pd.Series({"A": 0.2, "B": 0.2, "NEW": 0.1, "CASH": 0.5})
    pockets = pd.Series({"A": "coeur", "B": "coeur", "NEW": "coeur", "CASH": "defensif"})
    out = redistribute_missing(w, pd.Index(["A", "B", "CASH"]), pockets)
    assert out["CASH"] == pytest.approx(0.5)                 # le monétaire ne grossit pas
    assert out[["A", "B"]].tolist() == pytest.approx([0.25, 0.25])


def test_without_groups_weight_is_spread_over_everything():
    w = pd.Series({"A": 0.25, "NEW": 0.25, "CASH": 0.5})
    out = redistribute_missing(w, pd.Index(["A", "CASH"]))
    assert out["CASH"] == pytest.approx(2 / 3)


def test_group_with_no_survivor_is_spread_over_all():
    w = pd.Series({"A": 0.4, "SAT": 0.1, "CASH": 0.5})
    pockets = pd.Series({"A": "coeur", "SAT": "satellite", "CASH": "defensif"})
    out = redistribute_missing(w, pd.Index(["A", "CASH"]), pockets)
    assert out.sum() == pytest.approx(1.0)
    assert out["A"] / out["CASH"] == pytest.approx(0.4 / 0.5)


def test_stress_test_keeps_cash_share_when_a_stock_is_missing():
    # Krach de -40 % sur les actions ; NEW n'existe pas pendant la crise
    idx = pd.bdate_range("2008-01-01", periods=60)
    crash = np.linspace(100, 60, 60)
    prices = pd.DataFrame({"OLD": crash, "NEW": np.nan, "CASH": 100.0}, index=idx)
    w = pd.Series({"OLD": 0.25, "NEW": 0.25, "CASH": 0.5})
    pockets = pd.Series({"OLD": "coeur", "NEW": "coeur", "CASH": "defensif"})
    naive = stress_test(prices, w, "x", idx[0].date(), idx[-1].date())
    fixed = stress_test(prices, w, "x", idx[0].date(), idx[-1].date(), groups=pockets)
    # Rebalancé chaque jour à 50 % d'actions : perte = 1 - produit(1 + 0,5 x rendement action)
    r = pd.Series(crash).pct_change().dropna()
    assert fixed.max_drawdown == pytest.approx(1 - (1 + 0.5 * r).prod())
    assert naive.max_drawdown < fixed.max_drawdown                     # l'ancienne méthode sous-estimait
    assert fixed.weight_covered == pytest.approx(0.75)
