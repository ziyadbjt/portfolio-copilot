"""Chargement et validation de la politique d'investissement (config/policy.yaml).

La politique est lue une seule fois, validée, puis figée dans des dataclasses
immuables : aucun module ne peut la modifier en cours d'exécution.
Une politique incohérente fait échouer le programme dès le démarrage.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "policy.yaml"


class PolicyError(ValueError):
    """Politique absente, mal formée ou incohérente."""


@dataclass(frozen=True)
class Defensive:
    ticker: str
    max_weight: float


@dataclass(frozen=True)
class Allocation:
    """Répartition de la poche actions entre cœur et satellite."""
    core_target: float
    satellite_target: float
    satellite_max: float
    satellite_themes: Mapping[str, tuple[str, ...]]   # thème -> industries Yahoo


@dataclass(frozen=True)
class Limits:
    max_line_weight: float
    max_satellite_line_weight: float
    max_sector_weight: float
    max_country_weight: float
    max_foreign_currency_weight: float


@dataclass(frozen=True)
class StressScenario:
    name: str
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class RiskRules:
    max_drawdown: float
    mc_breach_probability_max: float
    mc_horizon_years: int
    mc_n_simulations: int
    var_confidence: float
    stress_scenarios: tuple[StressScenario, ...]


@dataclass(frozen=True)
class CoreStyle:
    min_dividend_yield: float
    max_payout_ratio: float


@dataclass(frozen=True)
class Screening:
    history_years: int
    volatility_years: int
    dividend_years: int
    dividend_cut_tolerance: float
    special_dividend_multiple: float
    max_dividend_cuts: int
    core_shortlist_size: int
    satellite_shortlist_size: int
    shortlist_max_per_sector: int
    core_score_weights: Mapping[str, float]
    satellite_score_weights: Mapping[str, float]


CORE_SCORE_CRITERIA = {"dividend_yield", "payout_ratio", "dividend_growth", "volatility"}
SATELLITE_SCORE_CRITERIA = {"volatility", "market_cap"}


@dataclass(frozen=True)
class Policy:
    base_currency: str
    horizon_years: int
    universe_regions: tuple[str, ...]
    defensive: Defensive
    allocation: Allocation
    limits: Limits
    risk: RiskRules
    core_style: CoreStyle
    screening: Screening
    excluded_industries: tuple[str, ...]


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> Policy:
    """Lit le fichier YAML et renvoie une Policy validée."""
    path = Path(path)
    if not path.exists():
        raise PolicyError(f"Fichier de politique introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_policy(raw)


def parse_policy(raw: dict) -> Policy:
    """Convertit un dictionnaire (YAML déjà lu) en Policy validée."""
    if not isinstance(raw, dict):
        raise PolicyError("La politique doit être un dictionnaire YAML.")

    defe = _section(raw, "defensive")
    alloc = _section(raw, "allocation")
    lim = _section(raw, "limits")
    risk = _section(raw, "risk")
    style = _section(raw, "core_style")
    excl = _section(raw, "exclusions")
    scr = _section(raw, "screening")

    policy = Policy(
        base_currency=str(_get(raw, "base_currency")),
        horizon_years=_positive_int(raw, "horizon_years"),
        universe_regions=tuple(_get(raw, "universe_regions")),
        defensive=Defensive(
            ticker=str(_get(defe, "ticker")),
            max_weight=_fraction(defe, "max_weight"),
        ),
        allocation=Allocation(
            core_target=_fraction(alloc, "core_target"),
            satellite_target=_fraction(alloc, "satellite_target"),
            satellite_max=_fraction(alloc, "satellite_max"),
            satellite_themes=_themes(_section(alloc, "satellite_themes")),
        ),
        limits=Limits(
            max_line_weight=_fraction(lim, "max_line_weight"),
            max_satellite_line_weight=_fraction(lim, "max_satellite_line_weight"),
            max_sector_weight=_fraction(lim, "max_sector_weight"),
            max_country_weight=_fraction(lim, "max_country_weight"),
            max_foreign_currency_weight=_fraction(lim, "max_foreign_currency_weight"),
        ),
        risk=RiskRules(
            max_drawdown=_fraction(risk, "max_drawdown"),
            mc_breach_probability_max=_fraction(risk, "mc_breach_probability_max"),
            mc_horizon_years=_positive_int(risk, "mc_horizon_years"),
            mc_n_simulations=_positive_int(risk, "mc_n_simulations"),
            var_confidence=_fraction(risk, "var_confidence"),
            stress_scenarios=_scenarios(_section(risk, "stress_scenarios")),
        ),
        core_style=CoreStyle(
            min_dividend_yield=_fraction(style, "min_dividend_yield"),
            max_payout_ratio=_fraction(style, "max_payout_ratio"),
        ),
        screening=Screening(
            history_years=_positive_int(scr, "history_years"),
            volatility_years=_positive_int(scr, "volatility_years"),
            dividend_years=_positive_int(scr, "dividend_years"),
            dividend_cut_tolerance=_fraction(scr, "dividend_cut_tolerance"),
            special_dividend_multiple=_multiple(scr, "special_dividend_multiple"),
            max_dividend_cuts=_non_negative_int(scr, "max_dividend_cuts"),
            core_shortlist_size=_positive_int(scr, "core_shortlist_size"),
            satellite_shortlist_size=_positive_int(scr, "satellite_shortlist_size"),
            shortlist_max_per_sector=_positive_int(scr, "shortlist_max_per_sector"),
            core_score_weights=_weights(scr, "core_score_weights", CORE_SCORE_CRITERIA),
            satellite_score_weights=_weights(
                scr, "satellite_score_weights", SATELLITE_SCORE_CRITERIA),
        ),
        excluded_industries=tuple(_get(excl, "industries")),
    )
    _check_consistency(policy)
    return policy


def _check_consistency(p: Policy) -> None:
    """Règles qui relient plusieurs champs entre eux."""
    a, lim = p.allocation, p.limits
    if abs(a.core_target + a.satellite_target - 1.0) > 1e-9:
        raise PolicyError("core_target + satellite_target doit faire 1.")
    if a.satellite_target > a.satellite_max:
        raise PolicyError("satellite_target ne peut pas dépasser satellite_max.")
    if lim.max_line_weight > lim.max_sector_weight:
        raise PolicyError("max_line_weight ne peut pas dépasser max_sector_weight.")
    if lim.max_satellite_line_weight > a.satellite_max:
        raise PolicyError("max_satellite_line_weight ne peut pas dépasser satellite_max.")
    s = p.screening
    # La short-list doit permettre d'investir chaque poche sans dépasser le poids max par ligne
    if s.core_shortlist_size * lim.max_line_weight < a.core_target - 1e-9:
        raise PolicyError("core_shortlist_size trop petit pour investir le cœur "
                          "avec max_line_weight.")
    if s.satellite_shortlist_size * lim.max_satellite_line_weight < a.satellite_target - 1e-9:
        raise PolicyError("satellite_shortlist_size trop petit pour investir le satellite "
                          "avec max_satellite_line_weight.")
    if max(s.volatility_years, s.dividend_years + 1) > s.history_years:
        raise PolicyError("history_years doit couvrir volatility_years et dividend_years + 1.")
    themed = [i for industries in a.satellite_themes.values() for i in industries]
    if set(themed) & set(p.excluded_industries):
        raise PolicyError("Une industrie ne peut pas être à la fois satellite et exclue.")
    if p.defensive.max_weight >= 1:
        raise PolicyError("defensive.max_weight doit rester sous 1 (sinon aucune action).")
    if not 0.5 <= p.risk.var_confidence < 1:
        raise PolicyError("var_confidence doit être entre 0.5 et 1 (ex. 0.95).")


# --- Lecture typée des champs -------------------------------------------------

def _get(d: dict, key: str):
    if key not in d or d[key] is None:
        raise PolicyError(f"Champ manquant : {key}")
    return d[key]


def _section(d: dict, key: str) -> dict:
    value = _get(d, key)
    if not isinstance(value, dict):
        raise PolicyError(f"La section {key} doit être un dictionnaire.")
    return value


def _fraction(d: dict, key: str) -> float:
    value = _get(d, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{key} doit être un nombre, reçu {value!r}.")
    if not 0 <= value <= 1:
        raise PolicyError(f"{key} doit être une fraction entre 0 et 1, reçu {value}.")
    return float(value)


def _positive_int(d: dict, key: str) -> int:
    value = _get(d, key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyError(f"{key} doit être un entier positif, reçu {value!r}.")
    return value


def _multiple(d: dict, key: str) -> float:
    value = _get(d, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 1:
        raise PolicyError(f"{key} doit être un nombre supérieur à 1, reçu {value!r}.")
    return float(value)


def _non_negative_int(d: dict, key: str) -> int:
    value = _get(d, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyError(f"{key} doit être un entier positif ou nul, reçu {value!r}.")
    return value


def _themes(raw: dict) -> Mapping[str, tuple[str, ...]]:
    if not raw:
        raise PolicyError("satellite_themes doit contenir au moins un thème.")
    for theme, industries in raw.items():
        if not isinstance(industries, list) or not industries:
            raise PolicyError(f"Le thème {theme} doit lister au moins une industrie.")
    # MappingProxyType : dictionnaire en lecture seule, frozen=True ne protège pas le contenu
    return MappingProxyType({theme: tuple(ind) for theme, ind in raw.items()})


def _weights(d: dict, key: str, allowed: set[str]) -> Mapping[str, float]:
    raw = _section(d, key)
    unknown = set(raw) - allowed
    if unknown:
        raise PolicyError(f"{key} : critères inconnus {sorted(unknown)} "
                          f"(autorisés : {sorted(allowed)}).")
    weights = {k: _fraction(raw, k) for k in raw}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise PolicyError(f"Les poids de {key} doivent faire 1.")
    return MappingProxyType(weights)


def _scenarios(raw: dict) -> tuple[StressScenario, ...]:
    scenarios = []
    for name, window in raw.items():
        start, end = _date(window, "start", name), _date(window, "end", name)
        if start >= end:
            raise PolicyError(f"Scénario {name} : start doit précéder end.")
        scenarios.append(StressScenario(name=name, start=start, end=end))
    if not scenarios:
        raise PolicyError("Au moins un scénario de stress est requis.")
    return tuple(scenarios)


def _date(window: dict, key: str, scenario: str) -> dt.date:
    value = window.get(key) if isinstance(window, dict) else None
    # PyYAML convertit déjà "2020-03-23" en date ; une faute de frappe reste une chaîne
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        raise PolicyError(f"Scénario {scenario} : date {key} invalide ({value!r}).") from None
