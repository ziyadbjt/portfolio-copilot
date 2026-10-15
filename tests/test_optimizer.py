"""Optimiseur et contrôles de politique sur un marché simulé au comportement connu."""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    InfeasibleError, equity_weights, optimize, shrunk_covariance, weekly_returns,
)
from src.policy_check import check_allocation, check_limits
from tests.conftest import with_cash


# --- Niveau 1 : poids des actions --------------------------------------------

def test_equity_weights_respect_every_limit(policy, market):
    prices, meta = market
    pol = with_cash(policy)
    eq = list(meta.index[:8])
    cov = shrunk_covariance(weekly_returns(prices[eq]))
    # 40 % d'actions : satellite = 8 % du total, possible avec 2 lignes à 5 % max
    w = equity_weights(cov, meta, pol, equity_share=0.4)
    assert w.sum() == pytest.approx(0.4)
    assert w[meta.loc[eq, "pocket"] == "coeur"].sum() == pytest.approx(0.4 * 0.8, abs=1e-4)
    assert w[meta.loc[eq, "pocket"] == "satellite"].sum() == pytest.approx(0.4 * 0.2, abs=1e-4)
    full = pd.concat([w, pd.Series({"CASH": 0.6})])
    failed = [c for c in check_limits(full, meta, pol) if not c.passed]
    assert failed == []


def test_equity_weights_infeasible_when_caps_too_tight(policy, market):
    prices, meta = market
    eq = list(meta.index[:8])
    cov = shrunk_covariance(weekly_returns(prices[eq]))
    # À 60 % d'actions, le satellite doit peser 12 % : 2 lignes à 5 % max ne suffisent pas
    with pytest.raises(InfeasibleError):
        equity_weights(cov, meta, with_cash(policy), equity_share=0.6)


def test_shrunk_covariance_is_symmetric_positive(market):
    prices, _ = market
    cov = shrunk_covariance(weekly_returns(prices.iloc[:, :8]))
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


# --- Contrôles de politique --------------------------------------------------

def test_check_limits_flags_each_breach(policy, market):
    _, meta = market
    w = pd.Series({"C1": 0.30, "C2": 0.10, "S1": 0.10, "CASH": 0.50})
    names = {c.name for c in check_limits(w, meta, with_cash(policy)) if not c.passed}
    assert "ligne C1" in names                 # 30 % > 10 %
    assert "ligne S1" in names                 # 10 % > 5 % (ligne satellite)
    assert "secteur Banks" in names            # 40 % > 30 %


def test_stress_check_rejects_full_equity(policy, market):
    prices, meta = market
    w = pd.Series(1 / 8, index=meta.index[:8])
    w = pd.concat([w, pd.Series({"CASH": 0.0})])
    report, details = check_allocation(w, meta, prices, with_cash(policy))
    stress_2008 = next(c for c in report.checks if c.name == "stress crise_2008")
    assert not stress_2008.passed
    assert stress_2008.value > 0.45            # krach simulé de -50 %


# --- Niveau 2 : part d'actions -----------------------------------------------

def test_optimize_finds_largest_accepted_equity_share(policy, market):
    prices, meta = market
    pol = with_cash(policy)
    res = optimize(prices, meta, pol)
    assert res.accepted
    stress = {c.name: c.value for c in res.report.checks if c.name.startswith("stress")}
    assert max(stress.values()) <= 0.20 + 1e-6
    # Un cran de plus d'actions doit échouer (sinon on n'a pas trouvé le maximum)
    refused_above = [t for t in res.trials if t.equity_share > res.equity_share]
    assert refused_above and not any(t.accepted for t in refused_above)
    # Ordre de grandeur : krach de ~-50 % x beta moyen ~1 -> environ 35-45 % d'actions
    assert 0.25 <= res.equity_share <= 0.50
    assert res.weights.sum() == pytest.approx(1.0)
    assert res.weights["CASH"] == pytest.approx(1 - res.equity_share)


def test_optimize_reports_failure_instead_of_hiding_it(policy, market):
    prices, meta = market
    from dataclasses import replace
    # Défensif plafonné à 10 % : au moins 90 % d'actions, intenable face au krach
    pol = with_cash(policy)
    pol = replace(pol, defensive=replace(pol.defensive, max_weight=0.10),
                  limits=replace(pol.limits, max_line_weight=0.30, max_sector_weight=0.60,
                                 max_country_weight=1.0, max_foreign_currency_weight=1.0,
                                 max_satellite_line_weight=0.15))
    res = optimize(prices, meta, pol)
    assert not res.accepted
    assert res.equity_share == pytest.approx(0.90)
    assert any(c.name == "stress crise_2008" for c in res.report.failed)



def test_weekly_returns_compound_daily_returns():
    idx = pd.bdate_range("2024-01-01", periods=10)          # lundi 1er au vendredi 12
    prices = pd.DataFrame({"X": [100, 101, 102, 103, 110, 111, 112, 113, 114, 121.0]}, index=idx)
    w = weekly_returns(prices)
    assert w["X"].tolist() == pytest.approx([121 / 110 - 1])  # vendredi -> vendredi


def test_weekly_returns_recover_correlation_hidden_by_time_zones():
    # Le même choc touche l'Europe un jour après les US : corrélation journalière ~0,
    # mais les deux bougent ensemble sur la semaine
    idx = pd.bdate_range("2010-01-01", periods=2600)
    shock = np.random.default_rng(9).normal(0, 0.01, len(idx))
    us = pd.Series(shock, index=idx)
    eu = us.shift(1).fillna(0)
    prices = pd.DataFrame({"US": 100 * (1 + us).cumprod(), "EU": 100 * (1 + eu).cumprod()})
    daily_corr = prices.pct_change().corr().loc["US", "EU"]
    weekly_corr = weekly_returns(prices).corr().loc["US", "EU"]
    assert abs(daily_corr) < 0.1
    assert weekly_corr > 0.7


def test_annualised_weekly_and_daily_covariances_agree_on_iid_data():
    idx = pd.bdate_range("2000-01-01", periods=5000)
    r = pd.DataFrame(np.random.default_rng(10).normal(0, 0.01, (5000, 2)), index=idx,
                     columns=["A", "B"])
    prices = 100 * (1 + r).cumprod()
    daily = shrunk_covariance(prices.pct_change().dropna(), frequency=252)
    weekly = shrunk_covariance(weekly_returns(prices), frequency=52)
    assert weekly.loc["A", "A"] == pytest.approx(daily.loc["A", "A"], rel=0.1)
