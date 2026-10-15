"""Mesures de risque d'un portefeuille. Code déterministe, sans réseau.

Conventions :
- rendements journaliers simples, prix en devise de base (le change est inclus) ;
- poids : pd.Series indexée par ticker, somme = 1 ;
- les pertes (VaR, CVaR, drawdown) sont renvoyées en nombres positifs : 0.20 = -20 % ;
- portefeuille rebalancé chaque jour à poids constants (simplification classique :
  les poids ne dérivent pas avec les prix).

Ces mesures estiment un risque à partir du passé. Elles ne garantissent rien.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


# --- Rendements --------------------------------------------------------------

def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements journaliers simples. La première ligne (sans veille) est retirée."""
    return prices.pct_change(fill_method=None).iloc[1:]


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Rendement journalier du portefeuille à poids constants.

    Seuls les jours où tous les titres pondérés ont un rendement sont gardés :
    un titre manquant ne doit pas être compté comme un rendement nul.
    """
    weights = _check_weights(weights)
    r = returns[weights.index].dropna()
    return r @ weights


def _check_weights(weights: pd.Series) -> pd.Series:
    if (weights < -1e-12).any():
        raise ValueError("Poids négatifs : l'outil ne gère pas la vente à découvert.")
    if abs(weights.sum() - 1) > 1e-6:
        raise ValueError(f"Les poids doivent faire 1, somme = {weights.sum():.4f}.")
    return weights[weights > 0]


# --- Volatilité et corrélations ----------------------------------------------

def annualized_volatility(returns: pd.Series | pd.DataFrame):
    """Écart-type des rendements journaliers x racine de 252."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Part de la variance du portefeuille due à chaque ligne (somme = 1).

    contribution_i = w_i x (Σ w)_i / (w' Σ w), avec Σ la matrice de covariance.
    Une ligne à 5 % du capital peut porter 15 % du risque : c'est ce qu'on veut voir.
    """
    weights = _check_weights(weights)
    cov = returns[weights.index].dropna().cov()
    marginal = cov @ weights
    return weights * marginal / (weights @ marginal)


# --- VaR, CVaR ---------------------------------------------------------------

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Perte journalière qui n'est dépassée que (1 - confidence) du temps, historiquement."""
    return float(-np.quantile(returns.dropna(), 1 - confidence))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall : perte moyenne les jours où la VaR est dépassée.

    La VaR dit "on perd au moins X les 5 % pires jours", la CVaR dit "combien en
    moyenne ces jours-là". Elle voit l'épaisseur de la queue que la VaR ignore.
    """
    r = returns.dropna()
    var = historical_var(r, confidence)
    return float(-r[r <= -var].mean())


def gaussian_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR sous hypothèse de loi normale, pour comparaison avec la VaR historique.

    Si elle est nettement plus faible que la VaR historique, les queues réelles
    sont plus épaisses que la normale : la normale sous-estime le risque.
    """
    r = returns.dropna()
    return float(-(r.mean() + norm.ppf(1 - confidence) * r.std()))


# --- Drawdown ----------------------------------------------------------------

def wealth_index(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Valeur du portefeuille au fil du temps, en partant de `start`."""
    return start * (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Baisse depuis le plus haut atteint, à chaque date (0 = au plus haut)."""
    wealth = wealth_index(returns)
    peak = np.maximum(wealth.cummax(), 1.0)      # le capital de départ compte comme plus haut
    return 1 - wealth / peak


def max_drawdown(returns: pd.Series) -> float:
    """Pire baisse depuis un plus haut sur la période (la mesure de la politique)."""
    if returns.empty:
        return 0.0
    return float(drawdown_series(returns).max())


# --- Concentration -----------------------------------------------------------

def weights_by_group(weights: pd.Series, groups: pd.Series) -> pd.Series:
    """Poids cumulé par groupe (secteur, pays, devise...), du plus gros au plus petit."""
    return weights.groupby(groups.reindex(weights.index)).sum().sort_values(ascending=False)


def effective_number_of_lines(weights: pd.Series) -> float:
    """1 / somme des poids au carré (inverse de l'indice de Herfindahl).

    10 lignes à 10 % donnent 10 ; 10 lignes dont une à 91 % donnent ~1,2.
    """
    return float(1 / (weights ** 2).sum())


def foreign_currency_weight(weights: pd.Series, currencies: pd.Series, base: str) -> float:
    """Part du portefeuille exposée à une devise autre que la devise de base."""
    ccy = currencies.reindex(weights.index)
    return float(weights[ccy != base].sum())


# --- Stress tests ------------------------------------------------------------

@dataclass(frozen=True)
class StressResult:
    scenario: str
    start: dt.date
    end: dt.date
    total_return: float        # rendement du portefeuille sur la fenêtre
    max_drawdown: float        # pire baisse dans la fenêtre
    weight_covered: float      # part du portefeuille qui existait déjà à l'époque


def redistribute_missing(
    weights: pd.Series, available: pd.Index, groups: pd.Series | None = None
) -> pd.Series:
    """Remet à 100 % un portefeuille dont certains titres n'ont pas de données.

    Le poids d'un titre absent va aux titres présents **du même groupe** (par
    exemple la même poche), au prorata de leurs poids. Sans cela, le poids d'une
    action absente irait en partie au monétaire et le portefeuille testé serait
    plus prudent que le vrai. Si aucun titre du groupe n'est présent, son poids
    est réparti sur tous les titres présents.
    """
    present = weights[weights.index.isin(available)]
    if present.empty:
        return present
    if groups is None:
        return present / present.sum()
    g = groups.reindex(weights.index)
    out = present.copy()
    orphan = 0.0
    for group, members in weights.groupby(g, dropna=False):
        lost = members[~members.index.isin(available)].sum()
        if lost == 0:
            continue
        kept = members[members.index.isin(available)]
        if kept.sum() > 0:
            out[kept.index] += lost * kept / kept.sum()
        else:
            orphan += lost
    if orphan:
        out += orphan * out / out.sum()
    return out / out.sum()


def stress_test(
    prices: pd.DataFrame,
    weights: pd.Series,
    name: str,
    start: dt.date,
    end: dt.date,
    groups: pd.Series | None = None,
) -> StressResult:
    """Rejoue le portefeuille actuel sur une crise passée.

    Un titre sans historique à l'époque (introduit en Bourse plus tard) est retiré
    et son poids redistribué dans son groupe (voir redistribute_missing).
    weight_covered dit quelle part du portefeuille le test couvre vraiment :
    sous ~80 %, le résultat est fragile.
    """
    weights = _check_weights(weights)
    window = prices.loc[pd.Timestamp(start):pd.Timestamp(end), weights.index]
    available = window.columns[window.notna().all()]
    covered = float(weights[available].sum())
    if covered == 0:
        return StressResult(name, start, end, np.nan, np.nan, 0.0)
    w = redistribute_missing(weights, available, groups)
    r = portfolio_returns(to_returns(window[w.index]), w)
    return StressResult(
        scenario=name, start=start, end=end,
        total_return=float((1 + r).prod() - 1),
        max_drawdown=max_drawdown(r),
        weight_covered=covered,
    )


# --- Monte Carlo -------------------------------------------------------------

@dataclass(frozen=True)
class MonteCarloResult:
    breach_probability: float  # proba que le max drawdown dépasse le seuil
    threshold: float
    median_max_drawdown: float
    p95_max_drawdown: float    # 5 % des scénarios font pire que ça
    median_final_value: float  # valeur finale médiane pour 1 investi
    n_simulations: int
    horizon_years: int


def simulate_max_drawdowns(
    returns: pd.Series,
    horizon_years: int,
    n_simulations: int = 10_000,
    block_size: int = 21,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simule l'horizon par bootstrap par blocs des rendements historiques du portefeuille.

    Renvoie (max drawdown de chaque scénario, valeur finale pour 1 investi).

    On tire au hasard des blocs de `block_size` jours consécutifs (~1 mois) dans
    l'historique et on les met bout à bout. Par rapport à une loi normale :
    - les queues épaisses réelles sont conservées (on rejoue de vrais jours) ;
    - les blocs gardent les périodes agitées groupées (volatilité en grappes),
      ce qui compte beaucoup pour le drawdown.
    Limite : on ne simule que des situations déjà vues dans l'historique.
    `seed` rend le résultat reproductible.
    """
    r = returns.dropna().to_numpy()
    n_days = horizon_years * TRADING_DAYS
    if len(r) < block_size:
        raise ValueError("Historique plus court qu'un bloc : simulation impossible.")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_days // block_size)              # arrondi au supérieur
    starts_max = len(r) - block_size + 1

    max_dds, finals = [], []
    for chunk in _chunks(n_simulations, 1000):       # par paquets pour limiter la mémoire
        starts = rng.integers(0, starts_max, size=(chunk, n_blocks))
        idx = (starts[:, :, None] + np.arange(block_size)).reshape(chunk, -1)[:, :n_days]
        wealth = np.cumprod(1 + r[idx], axis=1)
        peak = np.maximum(np.maximum.accumulate(wealth, axis=1), 1.0)
        max_dds.append((1 - wealth / peak).max(axis=1))
        finals.append(wealth[:, -1])
    return np.concatenate(max_dds), np.concatenate(finals)


def monte_carlo_drawdown(
    returns: pd.Series,
    horizon_years: int,
    threshold: float,
    n_simulations: int = 10_000,
    block_size: int = 21,
    seed: int = 42,
) -> MonteCarloResult:
    """Résumé de simulate_max_drawdowns : probabilité de dépasser `threshold`, quantiles."""
    max_dd, final = simulate_max_drawdowns(returns, horizon_years, n_simulations, block_size, seed)
    return MonteCarloResult(
        breach_probability=float((max_dd > threshold).mean()),
        threshold=threshold,
        median_max_drawdown=float(np.median(max_dd)),
        p95_max_drawdown=float(np.quantile(max_dd, 0.95)),
        median_final_value=float(np.median(final)),
        n_simulations=n_simulations,
        horizon_years=horizon_years,
    )


def _chunks(total: int, size: int):
    while total > 0:
        yield min(size, total)
        total -= size
