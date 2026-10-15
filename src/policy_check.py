"""Vérifie qu'une allocation respecte la politique. Un verdict par règle, chiffré.

Deux familles de contrôles :
- limites de construction (poids par ligne, secteur, pays, devise, poches) : rapides ;
- limites de risque (drawdown en stress test, probabilité Monte Carlo) : plus lentes.

Une allocation est acceptée seulement si tous les contrôles passent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import risk
from src.policy import Policy

DEFENSIVE = "defensif"
TOLERANCE = 1e-6        # marge numérique du solveur


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    limit: float
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class PolicyReport:
    checks: tuple[Check, ...]

    @property
    def accepted(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])


def _at_most(name: str, value: float, limit: float, detail: str = "") -> Check:
    return Check(name, float(value), float(limit), value <= limit + TOLERANCE, detail)


# --- Limites de construction -------------------------------------------------

def check_limits(weights: pd.Series, meta: pd.DataFrame, policy: Policy) -> list[Check]:
    """meta : index ticker, colonnes pocket (coeur / satellite / defensif), sector, country, currency."""
    lim, alloc = policy.limits, policy.allocation
    checks = [
        Check("somme des poids", float(weights.sum()), 1.0, abs(weights.sum() - 1) < TOLERANCE),
        Check("pas de vente à découvert", float(weights.min()), 0.0, weights.min() >= -TOLERANCE),
    ]
    pocket = meta["pocket"].reindex(weights.index)
    equity = weights[pocket != DEFENSIVE]
    defensive = float(weights[pocket == DEFENSIVE].sum())
    checks.append(_at_most("poche défensive", defensive, policy.defensive.max_weight))

    for ticker, w in weights[pocket == "coeur"].items():
        checks.append(_at_most(f"ligne {ticker}", w, lim.max_line_weight))
    for ticker, w in weights[pocket == "satellite"].items():
        checks.append(_at_most(f"ligne {ticker}", w, lim.max_satellite_line_weight))

    if equity.sum() > 0:
        sat_share = weights[pocket == "satellite"].sum() / equity.sum()
        checks.append(_at_most("satellite / actions", sat_share, alloc.satellite_max))

    # Secteur et pays : sur les actions (le monétaire n'a ni secteur ni pays)
    for label, column, limit in [("secteur", "sector", lim.max_sector_weight),
                                 ("pays", "country", lim.max_country_weight)]:
        for group, w in risk.weights_by_group(equity, meta[column]).items():
            checks.append(_at_most(f"{label} {group}", w, limit))

    fx = risk.foreign_currency_weight(weights, meta["currency"], policy.base_currency)
    checks.append(_at_most(f"hors {policy.base_currency}", fx, lim.max_foreign_currency_weight))
    return checks


# --- Limites de risque -------------------------------------------------------

def check_risk(
    weights: pd.Series, prices: pd.DataFrame, policy: Policy, seed: int = 42
) -> tuple[list[Check], dict]:
    """Stress tests et Monte Carlo. Renvoie aussi les résultats détaillés."""
    rules = policy.risk
    checks, details = [], {"stress": [], "monte_carlo": None}

    for s in rules.stress_scenarios:
        res = risk.stress_test(prices, weights, s.name, s.start, s.end)
        details["stress"].append(res)
        checks.append(_at_most(
            f"stress {s.name}", res.max_drawdown, rules.max_drawdown,
            f"couverture {res.weight_covered:.0%} du portefeuille",
        ))

    port = risk.portfolio_returns(risk.to_returns(prices), weights)
    mc = risk.monte_carlo_drawdown(
        port, rules.mc_horizon_years, rules.max_drawdown, rules.mc_n_simulations, seed=seed)
    details["monte_carlo"] = mc
    details["mc_history"] = (port.index.min(), port.index.max())
    checks.append(_at_most(
        f"P(drawdown > {rules.max_drawdown:.0%}) sur {rules.mc_horizon_years} ans",
        mc.breach_probability, rules.mc_breach_probability_max,
        f"historique du {port.index.min():%d/%m/%Y} au {port.index.max():%d/%m/%Y}",
    ))
    return checks, details


def check_allocation(
    weights: pd.Series, meta: pd.DataFrame, prices: pd.DataFrame, policy: Policy
) -> tuple[PolicyReport, dict]:
    limit_checks = check_limits(weights, meta, policy)
    risk_checks, details = check_risk(weights, prices, policy)
    return PolicyReport(tuple(limit_checks + risk_checks)), details
