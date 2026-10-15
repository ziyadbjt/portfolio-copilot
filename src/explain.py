"""Explication en français, générée par un gabarit (pas de LLM).

Deux temps, séparés exprès :
1. explanation_facts : extrait les chiffres et les constats (code déterministe) ;
2. render_explanation : un gabarit Jinja met ces faits en phrases.
Le texte ne peut donc contenir aucun chiffre qui ne vienne pas des calculs.
En phase 2, un agent pourra rédiger à partir des mêmes faits : on pourra comparer.
"""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader

from src.optimizer import OptimizationResult
from src.policy import Policy
from src.policy_check import DEFENSIVE
from src.report import TEMPLATES_DIR, Analytics

# En deçà, le résultat d'un stress test repose sur trop peu du portefeuille
LOW_COVERAGE = 0.9


def binding_constraints(result: OptimizationResult) -> list[str]:
    """Contrôles qui ont fait échouer la part d'actions juste au-dessus de celle retenue."""
    above = [t for t in result.trials if t.equity_share > result.equity_share and not t.accepted]
    if not above:
        return []
    closest = min(above, key=lambda t: t.equity_share)
    return [r.strip() for r in closest.reason.split(",") if r.strip()]


def explanation_facts(
    result: OptimizationResult, a: Analytics, policy: Policy, shortlist_validated: bool
) -> dict:
    w = a.weights
    equity = w[w["pocket"] != DEFENSIVE]
    worst = a.stress.loc[a.stress["max_drawdown"].idxmax()]
    low_cov = a.stress[a.stress["weight_covered"] < LOW_COVERAGE]
    mc = result.details["monte_carlo"]
    mc_start = result.details["mc_history"][0]
    risk_gap = (equity["risk_contribution"] - equity["weight"] / equity["weight"].sum())
    heaviest_risk = risk_gap.idxmax()

    return {
        "accepted": result.accepted,
        "validated": shortlist_validated,
        "equity_share": result.equity_share,
        "defensive_share": 1 - result.equity_share,
        "min_equity": 1 - policy.defensive.max_weight,
        "is_max_equity": result.equity_share >= 1 - 1e-9,
        "binding": binding_constraints(result),
        "failed": [c.name for c in result.report.failed],
        "n_lines": len(equity),
        "core_share": float(equity.loc[equity["pocket"] == "coeur", "weight"].sum()),
        "satellite_share": float(equity.loc[equity["pocket"] == "satellite", "weight"].sum()),
        "max_dd_limit": policy.risk.max_drawdown,
        "worst_scenario": worst["scenario"],
        "worst_drawdown": float(worst["max_drawdown"]),
        "low_coverage": [(r.scenario, r.weight_covered) for r in low_cov.itertuples()],
        "mc": mc,
        "mc_limit": policy.risk.mc_breach_probability_max,
        "mc_start": mc_start,
        "mc_misses_2008": mc_start.year > 2008,
        "history_start": a.metrics["start"],
        "vol": a.metrics["volatility"],
        "vol_equity": a.equity_only_metrics["volatility"],
        "mdd": a.metrics["max_drawdown"],
        "mdd_equity": a.equity_only_metrics["max_drawdown"],
        "var_99": a.metrics["var_99"],
        "gaussian_var_99": a.metrics["gaussian_var_99"],
        "top_line": equity["weight"].idxmax(),
        "top_line_name": equity.loc[equity["weight"].idxmax(), "name"],
        "top_line_weight": float(equity["weight"].max()),
        "top_sector": a.sectors.index[0],
        "top_sector_weight": float(a.sectors.iloc[0]),
        "top_country": a.countries.index[0],
        "top_country_weight": float(a.countries.iloc[0]),
        "heaviest_risk": heaviest_risk,
        "heaviest_risk_name": equity.loc[heaviest_risk, "name"],
        "heaviest_risk_weight": float(equity.loc[heaviest_risk, "weight"] / equity["weight"].sum()),
        "heaviest_risk_contrib": float(equity.loc[heaviest_risk, "risk_contribution"]),
        "base_currency": policy.base_currency,
        "horizon": policy.horizon_years,
    }


def render_explanation(facts: dict, template: str = "explanation.j2") -> list[str]:
    """Paragraphes de l'explication (séparés par une ligne vide dans le gabarit)."""
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), trim_blocks=True, lstrip_blocks=True)
    env.filters["pct"] = lambda x, d=0: f"{x:.{d}%}".replace(".", ",")
    text = env.get_template(template).render(**facts)
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    return [p for p in paragraphs if p]
