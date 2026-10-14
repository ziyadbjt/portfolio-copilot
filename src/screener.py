"""Screening : de l'univers à une short-list proposée, avec une raison pour chaque titre.

Étapes, toutes déterministes :
1. poche : l'industrie décide (thème IA / énergie -> satellite, sinon cœur) ;
2. filtres : exclusions éthiques, données suffisantes, puis critères dividendes (cœur) ;
3. score : moyenne pondérée de rangs percentiles, calculée dans chaque poche ;
4. short-list : meilleurs scores, avec un nombre max de titres par secteur.

Le résultat est une proposition : l'humain la relit et la valide.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.policy import Policy

TRADING_DAYS = 252

# Sens de chaque critère de score : True si une valeur basse est meilleure
LOWER_IS_BETTER = {
    "dividend_yield": False,
    "payout_ratio": True,
    "dividend_growth": False,
    "volatility": True,
    "market_cap": False,
}


# --- Mesures par titre -------------------------------------------------------

def assign_pocket(industry: pd.Series, themes: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    """Poche (coeur / satellite) et thème de chaque titre, selon son industrie."""
    theme_of = {ind: theme for theme, industries in themes.items() for ind in industries}
    theme = industry.map(theme_of)
    pocket = np.where(theme.notna(), "satellite", "coeur")
    return pd.DataFrame({"pocket": pocket, "theme": theme}, index=industry.index)


def special_dividend_mask(
    dividends: pd.DataFrame, multiple: float, window: str = "396D"
) -> pd.DataFrame:
    """Vrai pour un dividende exceptionnel : un pic qui ne se répète pas.

    Un versement est exceptionnel s'il :
    - dépasse `multiple` fois le plus gros versement des 13 mois précédents,
    - et n'est suivi, dans les 13 mois suivants, d'aucun versement d'au moins
      1/`multiple` de son montant (une vraie hausse de dividende, elle, se répète).
    Il faut 13 mois d'historique avant le versement pour juger ; un versement
    sans suite connue (trop récent) n'est jugé que sur le premier critère.
    13 mois et non 12 : la date de versement glisse de quelques jours selon les années.

    Pourquoi : un dividende exceptionnel (cession d'activité, excédent de
    trésorerie) gonflerait le rendement, puis ferait croire à une baisse du
    dividende l'année suivante.
    """
    dividends = dividends.sort_index()
    span = pd.Timedelta(window)
    data_start = dividends.index.min()
    mask = pd.DataFrame(False, index=dividends.index, columns=dividends.columns)
    for ticker in dividends.columns:
        paid = dividends[ticker][dividends[ticker] > 0]
        for date, amount in paid.items():
            if date - span < data_start:
                continue                                    # pas assez de recul
            before = paid[(paid.index >= date - span) & (paid.index < date)]
            after = paid[(paid.index > date) & (paid.index <= date + span)]
            spike = before.empty or amount > multiple * before.max()
            repeated = (after >= amount / multiple).any()
            mask.loc[date, ticker] = spike and not repeated
    return mask


def trailing_dividend_yield(
    dividends: pd.DataFrame, prices: pd.DataFrame, asof: pd.Timestamp
) -> pd.Series:
    """Dividendes versés sur les 12 derniers mois / dernier prix (même devise)."""
    paid = dividends.loc[asof - pd.DateOffset(years=1) + pd.Timedelta(days=1):asof].sum()
    last_price = prices.loc[:asof].ffill().iloc[-1]
    return paid / last_price


def annual_dividends(dividends: pd.DataFrame, last_year: int, n_years: int) -> pd.DataFrame:
    """Dividende par action de chaque année civile (lignes = années, colonnes = tickers)."""
    by_year = dividends.groupby(dividends.index.year).sum()
    years = range(last_year - n_years + 1, last_year + 1)
    return by_year.reindex(years, fill_value=0.0)


def count_dividend_cuts(annual: pd.DataFrame, tolerance: float) -> pd.Series:
    """Nombre d'années où le dividende baisse de plus de `tolerance` sur un an."""
    cuts = annual < annual.shift(1) * (1 - tolerance)
    return cuts.iloc[1:].sum().astype(int)


def dividend_growth(annual: pd.DataFrame) -> pd.Series:
    """Croissance annuelle moyenne (CAGR) du dividende entre la 1re et la dernière année."""
    first, last = annual.iloc[0], annual.iloc[-1]
    n = len(annual) - 1
    growth = (last / first) ** (1 / n) - 1
    return growth.where(first > 0)


def annualized_volatility(prices: pd.DataFrame, years: int, asof: pd.Timestamp) -> pd.Series:
    """Écart-type des rendements journaliers sur `years` ans, annualisé (x racine de 252)."""
    window = prices.loc[asof - pd.DateOffset(years=years):asof]
    returns = window.pct_change(fill_method=None)
    return returns.std() * np.sqrt(TRADING_DAYS)


# --- Score et short-list -----------------------------------------------------

def percentile_score(metrics: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Score entre 0 et 1 : moyenne pondérée des rangs percentiles de chaque critère.

    Des rangs plutôt que des valeurs brutes : les critères n'ont pas la même unité
    (un rendement en %, une capitalisation en milliards) et une valeur extrême ne
    peut pas écraser les autres critères.
    """
    score = pd.Series(0.0, index=metrics.index)
    for criterion, weight in weights.items():
        ranks = metrics[criterion].rank(pct=True, ascending=not LOWER_IS_BETTER[criterion])
        score += weight * ranks.fillna(0.0)    # critère inconnu = pire rang
    return score


def select_shortlist(candidates: pd.DataFrame, size: int, max_per_sector: int) -> list[str]:
    """Les `size` meilleurs scores, sans dépasser `max_per_sector` titres par secteur."""
    chosen, per_sector = [], {}
    for ticker, row in candidates.sort_values("score", ascending=False).iterrows():
        if per_sector.get(row["sector"], 0) >= max_per_sector:
            continue
        chosen.append(ticker)
        per_sector[row["sector"]] = per_sector.get(row["sector"], 0) + 1
        if len(chosen) == size:
            break
    return chosen


# --- Assemblage --------------------------------------------------------------

def screen(
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    dividends_local: pd.DataFrame,
    policy: Policy,
) -> pd.DataFrame:
    """Applique filtres et score à tout l'univers.

    universe : index ticker, colonnes name, country, sector, industry
    fundamentals : index ticker, colonnes payout_ratio, market_cap
    prices, dividends : en devise de base ; dividends_local : en devise de cotation

    Renvoie une ligne par ticker : mesures, poche, statut (shortlist / candidat /
    ecarte), raisons d'exclusion, score et rang dans la poche.
    """
    s, style = policy.screening, policy.core_style
    out = universe[["name", "country", "sector", "industry"]].copy()
    out = out.join(assign_pocket(out["industry"], policy.allocation.satellite_themes))
    out = out.join(fundamentals[["payout_ratio", "market_cap"]])

    has_prices = out.index.isin(prices.columns)
    priced = out.index[has_prices]
    asof = prices.index.max()
    special = special_dividend_mask(dividends_local[priced], s.special_dividend_multiple)
    regular_local = dividends_local[priced].mask(special, 0.0)
    regular = dividends[priced].mask(special, 0.0)
    annual = annual_dividends(regular_local, asof.year - 1, s.dividend_years)

    out["special_dividends"] = special.sum()
    out["dividend_yield"] = trailing_dividend_yield(regular, prices[priced], asof)
    out["dividend_growth"] = dividend_growth(annual)
    out["dividend_cuts"] = count_dividend_cuts(annual, s.dividend_cut_tolerance)
    out["years_without_dividend"] = (annual == 0).sum()
    out["volatility"] = annualized_volatility(prices[priced], s.volatility_years, asof)
    # Payout nul alors qu'un dividende est versé : bénéfice manquant chez Yahoo, donc inconnu
    out.loc[out["payout_ratio"] <= 0, "payout_ratio"] = np.nan

    reasons = {t: [] for t in out.index}

    def flag(mask: pd.Series, reason: str) -> None:
        for t in out.index[mask.fillna(False).to_numpy(dtype=bool)]:
            reasons[t].append(reason)

    flag(out["industry"].isin(policy.excluded_industries), "industrie exclue")
    flag(pd.Series(~has_prices, index=out.index), "historique de prix insuffisant")
    core = (out["pocket"] == "coeur") & has_prices
    flag(core & (out["dividend_yield"] < style.min_dividend_yield),
         f"rendement < {style.min_dividend_yield:.0%}")
    flag(core & out["payout_ratio"].isna(), "payout inconnu")
    flag(core & (out["payout_ratio"] > style.max_payout_ratio),
         f"payout > {style.max_payout_ratio:.0%}")
    flag(core & (out["years_without_dividend"] > 0),
         f"dividende non versé chaque année sur {s.dividend_years} ans")
    flag(core & (out["dividend_cuts"] > s.max_dividend_cuts),
         f"plus de {s.max_dividend_cuts} baisse(s) du dividende sur {s.dividend_years} ans")

    out["reasons"] = pd.Series({t: "; ".join(r) for t, r in reasons.items()})
    out["status"] = np.where(out["reasons"] == "", "candidat", "ecarte")
    out["score"] = np.nan
    out["rank"] = np.nan

    for pocket, weights, size in [
        ("coeur", s.core_score_weights, s.core_shortlist_size),
        ("satellite", s.satellite_score_weights, s.satellite_shortlist_size),
    ]:
        eligible = out[(out["pocket"] == pocket) & (out["status"] == "candidat")]
        if eligible.empty:
            continue
        scores = percentile_score(eligible, weights)
        out.loc[scores.index, "score"] = scores
        out.loc[scores.index, "rank"] = scores.rank(ascending=False, method="first")
        chosen = select_shortlist(out.loc[scores.index], size, s.shortlist_max_per_sector)
        out.loc[chosen, "status"] = "shortlist"

    status_order = {"shortlist": 0, "candidat": 1, "ecarte": 2}
    out["_order"] = out["status"].map(status_order)
    out = out.sort_values(["pocket", "_order", "score"], ascending=[True, True, False])
    return out.drop(columns="_order")
