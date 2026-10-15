"""Rapport : chiffres, graphiques matplotlib, page HTML (Jinja) et export Excel.

Ce module ne décide rien : il met en forme les résultats de l'optimiseur et des
mesures de risque. Tous les chiffres affichés sont calculés ici ou dans src/risk.py.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # rendu sans fenêtre (script, serveur)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from src import risk  # noqa: E402
from src.optimizer import OptimizationResult  # noqa: E402
from src.policy import Policy  # noqa: E402
from src.policy_check import DEFENSIVE  # noqa: E402

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

# Palette de référence (thème clair), voir README : couleurs par rôle, jamais par rang
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
POCKET_COLORS = {DEFENSIVE: "#2a78d6", "coeur": "#eb6834", "satellite": "#1baf7a"}
POCKET_LABELS = {DEFENSIVE: "Défensif (monétaire)", "coeur": "Cœur", "satellite": "Satellite"}
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
CRITICAL = "#d03b3b"
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3b3b2", "#e66767", "#a8302f"]


# --- Chiffres ----------------------------------------------------------------

@dataclass
class Analytics:
    weights: pd.DataFrame            # une ligne par titre : poids, poche, secteur, pays, devise...
    pockets: pd.Series
    sectors: pd.Series               # poids des actions par secteur (en part du total)
    countries: pd.Series
    currencies: pd.Series
    metrics: dict                    # volatilité, VaR, CVaR, drawdown... du portefeuille
    equity_only_metrics: dict        # mêmes mesures pour la poche actions seule (comparaison)
    portfolio_returns: pd.Series
    equity_only_returns: pd.Series
    risk_contrib: pd.Series
    correlations: pd.DataFrame
    stress: pd.DataFrame
    mc_max_drawdowns: np.ndarray
    checks: pd.DataFrame
    trials: pd.DataFrame


def _metrics(r: pd.Series, confidence: float) -> dict:
    return {
        "start": r.index.min(), "end": r.index.max(), "days": len(r),
        "annual_return": float((1 + r).prod() ** (risk.TRADING_DAYS / len(r)) - 1),
        "volatility": float(risk.annualized_volatility(r)),
        "var": risk.historical_var(r, confidence),
        "cvar": risk.historical_cvar(r, confidence),
        "gaussian_var": risk.gaussian_var(r, confidence),
        "var_99": risk.historical_var(r, 0.99),
        "gaussian_var_99": risk.gaussian_var(r, 0.99),
        "max_drawdown": risk.max_drawdown(r),
    }


def build_analytics(
    result: OptimizationResult,
    prices: pd.DataFrame,
    meta: pd.DataFrame,
    names: pd.Series,
    policy: Policy,
) -> Analytics:
    w = result.weights[result.weights > 0].sort_values(ascending=False)
    pocket = meta["pocket"].reindex(w.index)
    equity = w[pocket != DEFENSIVE]
    returns = risk.to_returns(prices)

    table = pd.DataFrame({"weight": w}).join(meta).join(names.rename("name"))
    contrib = risk.risk_contributions(returns.dropna(subset=list(equity.index)), w)
    table["risk_contribution"] = contrib
    table["pocket_label"] = table["pocket"].map(POCKET_LABELS)
    table[["name", "sector", "country"]] = table[["name", "sector", "country"]].fillna("")

    port = risk.portfolio_returns(returns, w)
    eq_only = risk.portfolio_returns(returns, equity / equity.sum())
    conf = policy.risk.var_confidence

    mc = result.details["monte_carlo"]
    max_dds, _ = risk.simulate_max_drawdowns(port, mc.horizon_years, mc.n_simulations)

    stress = pd.DataFrame([s.__dict__ for s in result.details["stress"]])
    trials = pd.DataFrame([t.__dict__ for t in result.trials]).sort_values("equity_share")

    return Analytics(
        weights=table,
        pockets=w.groupby(pocket).sum(),
        sectors=risk.weights_by_group(equity, meta["sector"]),
        countries=risk.weights_by_group(equity, meta["country"]),
        currencies=risk.weights_by_group(w, meta["currency"]),
        metrics=_metrics(port, conf),
        equity_only_metrics=_metrics(eq_only, conf),
        portfolio_returns=port,
        equity_only_returns=eq_only,
        risk_contrib=contrib,
        correlations=risk.correlation_matrix(returns[list(equity.index)].dropna()),
        stress=stress,
        mc_max_drawdowns=max_dds,
        checks=result.report.to_frame(),
        trials=trials,
    )


# --- Graphiques --------------------------------------------------------------

def _style(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _pct(ax, axis: str = "x", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _new(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def chart_pockets(a: Analytics) -> str:
    """Une barre à 100 % : défensif / cœur / satellite, avec étiquettes directes."""
    fig, ax = _new(8, 1.3)
    left = 0.0
    for pocket in (DEFENSIVE, "coeur", "satellite"):
        value = float(a.pockets.get(pocket, 0.0))
        if value <= 0:
            continue
        ax.barh(0, value, left=left, height=0.5, color=POCKET_COLORS[pocket],
                edgecolor=SURFACE, linewidth=2)
        ax.text(left + value / 2, 0.42, f"{POCKET_LABELS[pocket]}  {value:.0%}",
                ha="center", va="bottom", fontsize=9, color=INK)
        left += value
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.9)
    ax.axis("off")
    return _to_base64(fig)


def chart_weights(a: Analytics, policy: Policy) -> str:
    """Poids de chaque ligne (actions), couleur = poche, avec les plafonds de la politique."""
    t = a.weights[a.weights["pocket"] != DEFENSIVE].sort_values("weight")
    fig, ax = _new(8, 0.28 * len(t) + 1)
    colors = [POCKET_COLORS[p] for p in t["pocket"]]
    ax.barh(t.index, t["weight"], color=colors, height=0.6)
    ax.axvline(policy.limits.max_line_weight, color=MUTED, linewidth=1)
    ax.text(policy.limits.max_line_weight, len(t) - 0.4, " plafond cœur", color=MUTED, fontsize=8)
    ax.axvline(policy.limits.max_satellite_line_weight, color=MUTED, linewidth=1)
    ax.text(policy.limits.max_satellite_line_weight, -0.9, " plafond satellite",
            color=MUTED, fontsize=8)
    for y, value in enumerate(t["weight"]):
        ax.text(value, y, f" {value:.1%}", va="center", fontsize=8, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=POCKET_COLORS[p]) for p in ("coeur", "satellite")]
    ax.legend(handles, [POCKET_LABELS["coeur"], POCKET_LABELS["satellite"]],
              frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    _pct(ax)
    return _to_base64(fig)


def chart_risk_vs_weight(a: Analytics) -> str:
    """Part du capital vs part du risque, ligne par ligne (actions)."""
    t = a.weights[a.weights["pocket"] != DEFENSIVE].sort_values("risk_contribution")
    y = np.arange(len(t))
    fig, ax = _new(8, 0.3 * len(t) + 1)
    ax.barh(y + 0.2, t["weight"] / t["weight"].sum(), height=0.38, color=BASELINE,
            label="part des actions")
    ax.barh(y - 0.2, t["risk_contribution"], height=0.38, color=SERIES_1,
            label="part du risque")
    ax.set_yticks(y, t.index)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    _pct(ax)
    return _to_base64(fig)


def chart_drawdown(a: Analytics, policy: Policy) -> str:
    """Historique de drawdown : portefeuille vs poche actions seule, limite à -20 %."""
    fig, ax = _new(8, 3)
    port = risk.drawdown_series(a.portfolio_returns)
    eq = risk.drawdown_series(a.equity_only_returns)
    ax.plot(eq.index, -eq, color=SERIES_2, linewidth=1.5, label="actions seules (100 %)")
    ax.plot(port.index, -port, color=SERIES_1, linewidth=2, label="portefeuille proposé")
    ax.axhline(-policy.risk.max_drawdown, color=CRITICAL, linewidth=1)
    ax.text(port.index[0], -policy.risk.max_drawdown, f"limite -{policy.risk.max_drawdown:.0%}",
            color=INK_2, fontsize=8, va="bottom")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax, "y")
    _pct(ax, "y")
    return _to_base64(fig)


def chart_wealth(a: Analytics) -> str:
    fig, ax = _new(8, 3)
    ax.plot(a.equity_only_returns.index, risk.wealth_index(a.equity_only_returns, 100),
            color=SERIES_2, linewidth=1.5, label="actions seules (100 %)")
    ax.plot(a.portfolio_returns.index, risk.wealth_index(a.portfolio_returns, 100),
            color=SERIES_1, linewidth=2, label="portefeuille proposé")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "y")
    return _to_base64(fig)


def chart_stress(a: Analytics, policy: Policy) -> str:
    """Drawdown de chaque scénario de crise vs la limite."""
    s = a.stress
    fig, ax = _new(8, 2.4)
    y = np.arange(len(s))
    ax.barh(y, s["max_drawdown"], height=0.5, color=SERIES_1)
    ax.axvline(policy.risk.max_drawdown, color=CRITICAL, linewidth=1)
    for i, row in s.iterrows():
        ok = row["max_drawdown"] <= policy.risk.max_drawdown + 1e-9
        ax.text(row["max_drawdown"] + 0.004, i, f" -{row['max_drawdown']:.1%}  {'✓' if ok else '✗ dépasse'} ",
                va="center", fontsize=9, color=INK_2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_yticks(y, s["scenario"])
    ax.set_xlim(0, max(policy.risk.max_drawdown, s["max_drawdown"].max()) * 1.35)
    _style(ax)
    _pct(ax)
    return _to_base64(fig)


def chart_monte_carlo(a: Analytics, policy: Policy) -> str:
    """Distribution du pire drawdown sur l'horizon, dans les scénarios simulés."""
    dd = a.mc_max_drawdowns
    threshold = policy.risk.max_drawdown
    fig, ax = _new(8, 3)
    bins = np.linspace(0, max(dd.max(), threshold * 1.5), 50)
    ax.hist(dd[dd <= threshold], bins=bins, color=SERIES_1, edgecolor=SURFACE, linewidth=0.5)
    ax.hist(dd[dd > threshold], bins=bins, color=CRITICAL, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(threshold, color=INK_2, linewidth=1)
    share = (dd > threshold).mean()
    ax.text(threshold, ax.get_ylim()[1] * 0.9,
            f"  {share:.1%} des scénarios dépassent -{threshold:.0%}", fontsize=9, color=INK)
    ax.set_ylabel("scénarios", color=MUTED, fontsize=9)
    _style(ax, "y")
    _pct(ax)
    return _to_base64(fig)


def chart_groups(series: pd.Series, limit: float, title: str) -> str:
    """Poids par groupe (secteur ou pays) en part du total, avec la limite."""
    s = series.sort_values()
    fig, ax = _new(4, 0.3 * len(s) + 1)
    ax.barh(s.index, s.values, height=0.6, color=SERIES_1)
    ax.axvline(limit, color=CRITICAL, linewidth=1)
    ax.text(limit, len(s) - 0.5, f" limite {limit:.0%}", fontsize=8, color=INK_2)
    for y, value in enumerate(s.values):
        ax.text(value, y, f" {value:.1%}", va="center", fontsize=8, color=INK_2)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlim(0, limit * 1.3)
    _style(ax)
    _pct(ax)
    return _to_base64(fig)


def chart_correlations(a: Analytics) -> str:
    """Matrice de corrélations des actions, échelle divergente centrée sur 0."""
    c = a.correlations
    fig, ax = _new(7, 6)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", DIVERGING)
    im = ax.imshow(c.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=7, color=INK_2)
    ax.set_yticks(range(len(c)), c.index, fontsize=7, color=INK_2)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    return _to_base64(fig)


def make_charts(a: Analytics, policy: Policy) -> dict[str, str]:
    return {
        "pockets": chart_pockets(a),
        "weights": chart_weights(a, policy),
        "risk_vs_weight": chart_risk_vs_weight(a),
        "drawdown": chart_drawdown(a, policy),
        "wealth": chart_wealth(a),
        "stress": chart_stress(a, policy),
        "monte_carlo": chart_monte_carlo(a, policy),
        "sectors": chart_groups(a.sectors, policy.limits.max_sector_weight, "Secteurs (actions)"),
        "countries": chart_groups(a.countries, policy.limits.max_country_weight, "Pays (actions)"),
        "correlations": chart_correlations(a),
    }


# --- Sorties -----------------------------------------------------------------

def render_html(context: dict, template: str = "report.html.j2") -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR),
                      autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = lambda x, d=1: "" if x is None or pd.isna(x) else f"{x:.{d}%}"
    env.filters["date"] = lambda x: "" if x is None else pd.Timestamp(x).strftime("%d/%m/%Y")
    return env.get_template(template).render(**context)


def write_excel(a: Analytics, path: Path) -> None:
    with pd.ExcelWriter(path) as xl:
        cols = ["name", "pocket", "sector", "country", "currency", "weight", "risk_contribution"]
        a.weights[cols].rename(columns={
            "name": "Nom", "pocket": "Poche", "sector": "Secteur", "country": "Pays",
            "currency": "Devise", "weight": "Poids", "risk_contribution": "Part du risque",
        }).to_excel(xl, sheet_name="Allocation")
        a.checks.rename(columns={"name": "Contrôle", "value": "Valeur", "limit": "Limite",
                                 "passed": "Respecté", "detail": "Détail"}) \
            .to_excel(xl, sheet_name="Contrôles", index=False)
        pd.DataFrame({"Portefeuille": a.metrics, "Actions seules": a.equity_only_metrics}) \
            .to_excel(xl, sheet_name="Mesures de risque")
        a.stress.to_excel(xl, sheet_name="Stress tests", index=False)
        a.trials.rename(columns={"equity_share": "Part d'actions", "accepted": "Acceptée",
                                 "reason": "Motif"}) \
            .to_excel(xl, sheet_name="Essais optimiseur", index=False)
        a.correlations.to_excel(xl, sheet_name="Corrélations")


def generated_at() -> str:
    return dt.datetime.now().strftime("%d/%m/%Y %H:%M")
