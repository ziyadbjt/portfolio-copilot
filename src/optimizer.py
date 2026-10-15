"""Optimisation sous contraintes : part d'actions, puis poids de chaque ligne.

Deux niveaux :
1. à part d'actions fixée, on cherche les poids des actions qui minimisent la
   volatilité, sous toutes les limites de la politique (ligne, secteur, pays,
   devise, répartition cœur / satellite) ;
2. on cherche la plus grande part d'actions dont l'allocation passe les limites
   de risque (stress tests et Monte Carlo). Le reste va dans la poche défensive.

Pourquoi minimiser la volatilité plutôt que maximiser un ratio de Sharpe :
le Sharpe exige des rendements attendus, très mal estimés à partir du passé ;
l'optimiseur surpondérerait alors ce qui a le plus monté. La covariance est
beaucoup plus stable, et le risque est justement ce que la politique contraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.exceptions import OptimizationError
from pypfopt.risk_models import CovarianceShrinkage

from src.policy import Policy
from src.policy_check import DEFENSIVE, PolicyReport, check_allocation


class InfeasibleError(RuntimeError):
    """Aucune allocation ne respecte les limites de construction."""


# --- Covariance --------------------------------------------------------------

def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements hebdomadaires (du vendredi au vendredi, dernier prix connu de la semaine).

    Paris ferme à 17h30, New York à 22h : une nouvelle tombée l'après-midi touche
    Paris le lendemain mais New York le jour même. En données journalières, les
    corrélations Europe / US paraissent donc plus faibles qu'elles ne sont
    (0,20 contre 0,26 en hebdomadaire sur la short-list) et l'optimiseur croit à
    une diversification qui n'existe pas. Sur une semaine, ce décalage disparaît.
    """
    return prices.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:]


def shrunk_covariance(returns: pd.DataFrame, frequency: int = 52) -> pd.DataFrame:
    """Covariance annualisée, rétrécie vers une cible simple (Ledoit-Wolf).

    Avec ~20 titres et quelques centaines de semaines, la covariance brute contient
    du bruit que l'optimiseur exploite comme s'il était réel (corrélations
    faibles par hasard -> poids extrêmes). Le shrinkage tire les estimations vers
    une structure plus stable, ce qui rend les poids moins fragiles.
    frequency : nombre de périodes par an (52 en hebdomadaire, 252 en journalier).
    """
    clean = returns.dropna()
    if len(clean) < 2 * returns.shape[1]:
        raise ValueError("Pas assez de périodes communes pour estimer la covariance.")
    return CovarianceShrinkage(clean, returns_data=True, frequency=frequency).ledoit_wolf()


# --- Niveau 1 : poids des actions à part d'actions fixée ----------------------

def equity_weights(
    cov: pd.DataFrame, meta: pd.DataFrame, policy: Policy, equity_share: float
) -> pd.Series:
    """Poids des actions (somme = equity_share) à volatilité minimale sous contraintes.

    Les limites de la politique portent sur le portefeuille total : à l'intérieur
    de la poche actions, une limite L devient L / equity_share.
    """
    tickers = list(cov.index)
    m = meta.reindex(tickers)
    e, lim, alloc = equity_share, policy.limits, policy.allocation

    is_core = (m["pocket"] == "coeur").to_numpy(dtype=float)
    is_sat = (m["pocket"] == "satellite").to_numpy(dtype=float)
    caps = np.where(is_core == 1, lim.max_line_weight, lim.max_satellite_line_weight) / e
    bounds = [(0.0, float(min(c, 1.0))) for c in caps]

    ef = EfficientFrontier(None, cov, weight_bounds=bounds)
    ef.add_objective(objective_functions.L2_reg, gamma=policy.optimizer.l2_gamma)
    ef.add_constraint(lambda w: is_core @ w == alloc.core_target)
    ef.add_constraint(lambda w: is_sat @ w == alloc.satellite_target)
    for column, limit in [("sector", lim.max_sector_weight), ("country", lim.max_country_weight)]:
        for group in m[column].dropna().unique():
            mask = (m[column] == group).to_numpy(dtype=float)
            ef.add_constraint(lambda w, mask=mask: mask @ w <= limit / e)
    foreign = (m["currency"] != policy.base_currency).to_numpy(dtype=float)
    ef.add_constraint(lambda w: foreign @ w <= lim.max_foreign_currency_weight / e)

    try:
        ef.min_volatility()
    except OptimizationError as err:
        raise InfeasibleError(f"Pas de solution à {e:.0%} d'actions : {err}") from err
    w = pd.Series(ef.clean_weights(cutoff=1e-4), dtype=float)
    return w / w.sum() * e


def full_allocation(equity: pd.Series, defensive_ticker: str) -> pd.Series:
    """Ajoute la poche défensive pour que le total fasse 100 %."""
    cash = 1.0 - equity.sum()
    return pd.concat([equity, pd.Series({defensive_ticker: cash})])


# --- Niveau 2 : part d'actions maximale acceptée ------------------------------

@dataclass
class Trial:
    equity_share: float
    accepted: bool
    reason: str


@dataclass
class OptimizationResult:
    weights: pd.Series
    equity_share: float
    report: PolicyReport
    details: dict
    trials: list[Trial] = field(default_factory=list)
    cov: pd.DataFrame | None = None

    @property
    def accepted(self) -> bool:
        return self.report.accepted


def optimize(prices: pd.DataFrame, meta: pd.DataFrame, policy: Policy) -> OptimizationResult:
    """Cherche par dichotomie la plus grande part d'actions acceptée.

    prices : prix en devise de base, poche défensive comprise (colonne policy.defensive.ticker)
    meta : index ticker, colonnes pocket, sector, country, currency (défensif compris)

    Hypothèse : plus il y a d'actions, plus le risque monte (vrai en pratique, le
    monétaire étant quasi sans risque). La dichotomie teste ~7 parts au lieu de 80.
    Si même la part d'actions minimale échoue, le résultat est renvoyé avec
    accepted = False : c'est à l'humain de revoir la politique ou la short-list.
    """
    defensive = policy.defensive.ticker
    equity_tickers = [t for t in meta.index if meta.at[t, "pocket"] != DEFENSIVE]
    window_start = prices.index.max() - pd.DateOffset(years=policy.optimizer.covariance_years)
    cov = shrunk_covariance(weekly_returns(prices.loc[window_start:, equity_tickers]))

    step = policy.optimizer.equity_step
    lo, hi = 1.0 - policy.defensive.max_weight, 1.0
    trials: list[Trial] = []
    cache: dict[float, tuple[pd.Series, PolicyReport, dict]] = {}

    def evaluate(e: float):
        e = round(e, 6)
        if e not in cache:
            try:
                w = full_allocation(equity_weights(cov, meta, policy, e), defensive)
            except InfeasibleError as err:
                trials.append(Trial(e, False, str(err)))
                cache[e] = (None, None, None)
                return cache[e]
            report, details = check_allocation(w, meta, prices, policy)
            reason = "acceptée" if report.accepted else ", ".join(c.name for c in report.failed)
            trials.append(Trial(e, report.accepted, reason))
            cache[e] = (w, report, details)
        return cache[e]

    w, report, details = evaluate(lo)
    if w is None or not report.accepted:
        if w is None:
            raise InfeasibleError(
                f"Même à {lo:.0%} d'actions, les limites de construction sont impossibles "
                "à respecter avec cette short-list.")
        return OptimizationResult(w, lo, report, details, trials, cov)

    best = lo
    if evaluate(hi)[1] is not None and evaluate(hi)[1].accepted:
        best = hi
    else:
        while hi - lo > step + 1e-9:
            mid = round((lo + hi) / 2 / step) * step
            if mid in (lo, hi):
                break
            w_mid, rep_mid, _ = evaluate(mid)
            if rep_mid is not None and rep_mid.accepted:
                lo = best = mid
            else:
                hi = mid

    w, report, details = evaluate(best)
    return OptimizationResult(w, best, report, details, trials, cov)

