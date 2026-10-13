"""Chargement et validation de la politique d'investissement (config/policy.yaml).

La politique est lue une seule fois, validée, puis figée dans des dataclasses
immuables : aucun module ne peut la modifier en cours d'exécution.
Une politique incohérente fait échouer le programme dès le démarrage.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "policy.yaml"


class PolicyError(ValueError):
    """Politique absente, mal formée ou incohérente."""


@dataclass(frozen=True)
class Allocation:
    core_target: float
    satellite_target: float
    satellite_max: float
    satellite_themes: tuple[str, ...]


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
class Policy:
    base_currency: str
    horizon_years: int
    universe_regions: tuple[str, ...]
    allocation: Allocation
    limits: Limits
    risk: RiskRules
    core_style: CoreStyle
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

    alloc = _section(raw, "allocation")
    lim = _section(raw, "limits")
    risk = _section(raw, "risk")
    style = _section(raw, "core_style")
    excl = _section(raw, "exclusions")

    policy = Policy(
        base_currency=str(_get(raw, "base_currency")),
        horizon_years=_positive_int(raw, "horizon_years"),
        universe_regions=tuple(_get(raw, "universe_regions")),
        allocation=Allocation(
            core_target=_fraction(alloc, "core_target"),
            satellite_target=_fraction(alloc, "satellite_target"),
            satellite_max=_fraction(alloc, "satellite_max"),
            satellite_themes=tuple(_get(alloc, "satellite_themes")),
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
