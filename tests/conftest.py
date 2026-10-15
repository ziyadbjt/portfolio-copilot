"""Fixtures partagées : politique allégée et marché simulé au comportement connu."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.policy import DEFAULT_POLICY_PATH, parse_policy


@pytest.fixture
def policy():
    with DEFAULT_POLICY_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["risk"]["mc_n_simulations"] = 400       # tests rapides
    return parse_policy(raw)


@pytest.fixture
def market():
    """2007-2024 : un facteur marché qui perd ~50 % pendant la fenêtre 2008, puis remonte.

    8 actions (6 cœur, 2 satellite) + un monétaire sans risque.
    """
    idx = pd.bdate_range("2007-01-01", "2024-12-31")
    rng = np.random.default_rng(0)
    factor = rng.normal(0.0004, 0.008, len(idx))
    crash = (idx >= "2007-10-09") & (idx <= "2009-03-09")
    factor[crash] = np.log(0.5) / crash.sum()                  # -50 % réguliers
    tickers = ["C1", "C2", "C3", "C4", "C5", "C6", "S1", "S2"]
    betas = [0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.4, 1.5]
    prices = pd.DataFrame({
        t: 100 * np.exp(np.cumsum(b * factor + rng.normal(0, 0.006, len(idx))))
        for t, b in zip(tickers, betas)
    }, index=idx)
    prices["CASH"] = 100 * np.exp(np.cumsum(np.full(len(idx), 0.00005)))
    meta = pd.DataFrame({
        "pocket": ["coeur"] * 6 + ["satellite"] * 2 + ["defensif"],
        "sector": ["Banks", "Banks", "Utilities", "Industrials", "Telecom", "Retail",
                   "Technology", "Energy", None],
        "country": ["France", "Germany", "France", "Spain", "United States", "Italy",
                    "United States", "United Kingdom", None],
        "currency": ["EUR", "EUR", "EUR", "EUR", "USD", "EUR", "USD", "GBP", "EUR"],
    }, index=tickers + ["CASH"])
    return prices, meta


def with_cash(policy, ticker="CASH"):
    """La politique réelle pointe sur XEON.DE : on la redirige vers le monétaire simulé."""
    return replace(policy, defensive=replace(policy.defensive, ticker=ticker))
