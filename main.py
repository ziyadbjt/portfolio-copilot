"""Phase 1 : de la short-list au rapport d'allocation.

Enchaîne : politique -> short-list -> données -> optimisation -> contrôles -> rapport.
Usage : python main.py   (ou cellule par cellule dans VS Code avec les blocs # %%)

Sorties : output/rapport.html et output/allocation.xlsx.
Lecture seule : rien n'est jamais acheté ni vendu.
"""

# %% Imports et chemins
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import report, risk  # noqa: E402
from src.data import fill_before_listing, load_market_data  # noqa: E402
from src.optimizer import optimize  # noqa: E402
from src.policy import Policy, load_policy  # noqa: E402
from src.policy_check import DEFENSIVE  # noqa: E402

VALIDATED_SHORTLIST = ROOT / "config" / "shortlist.csv"
PROPOSED_SHORTLIST = ROOT / "output" / "shortlist_proposee.csv"
UNIVERSE = ROOT / "config" / "universe.csv"
OUTPUT = ROOT / "output"


# %% Étapes
def load_shortlist() -> tuple[pd.DataFrame, bool]:
    """Short-list validée par l'humain si elle existe, sinon la proposition (signalée)."""
    if VALIDATED_SHORTLIST.exists():
        return pd.read_csv(VALIDATED_SHORTLIST, index_col="ticker"), True
    if PROPOSED_SHORTLIST.exists():
        print("Attention : short-list non validée, utilisation de la proposition du screening.")
        return pd.read_csv(PROPOSED_SHORTLIST, index_col="ticker"), False
    sys.exit("Aucune short-list : lancer d'abord scripts/run_screening.py")


def build_meta(shortlist: pd.DataFrame, currencies: dict, policy: Policy) -> pd.DataFrame:
    """Poche, secteur, pays et devise de chaque ligne, poche défensive comprise."""
    universe = pd.read_csv(UNIVERSE, index_col="ticker")
    meta = shortlist[["pocket"]].join(universe[["sector", "country"]])
    meta["currency"] = pd.Series(currencies)
    meta.loc[policy.defensive.ticker] = [DEFENSIVE, None, None, policy.base_currency]
    return meta


def names_of(shortlist: pd.DataFrame, policy: Policy) -> pd.Series:
    names = shortlist["name"].copy()
    names[policy.defensive.ticker] = "Fonds monétaire EUR (€STR)"
    return names


def run() -> dict:
    policy = load_policy()
    shortlist, validated = load_shortlist()
    tickers = list(shortlist.index) + [policy.defensive.ticker]

    print(f"Chargement de {len(tickers)} titres depuis {policy.optimizer.history_start}...")
    md = load_market_data(tickers, policy.optimizer.history_start, dt.date.today(),
                          policy.base_currency, min_coverage=0)
    if policy.defensive.ticker not in md.prices.columns:
        sys.exit(f"Poche défensive {policy.defensive.ticker} introuvable : {md.dropped}")
    prices = fill_before_listing(md.prices, policy.defensive.ticker)
    meta = build_meta(shortlist.loc[shortlist.index.isin(prices.columns)], md.currencies, policy)

    print("Optimisation...")
    result = optimize(prices, meta, policy)
    print(f"  part d'actions : {result.equity_share:.0%} · "
          f"{'acceptée' if result.accepted else 'NON CONFORME'}")

    analytics = report.build_analytics(result, prices, meta, names_of(shortlist, policy), policy)
    # Garde-fou : le Monte Carlo du rapport doit être celui qui a servi à décider
    replay = (analytics.mc_max_drawdowns > policy.risk.max_drawdown).mean()
    assert np.isclose(replay, result.details["monte_carlo"].breach_probability), \
        "Monte Carlo du rapport différent de celui des contrôles"

    data_notes = [f"{t} écarté : {reason}" for t, reason in md.dropped.items()]
    if not md.suspicious.empty:
        data_notes.append(f"{len(md.suspicious)} variations journalières de plus de 50 % "
                          "signalées dans les données (à vérifier).")

    equity = result.weights[meta["pocket"].reindex(result.weights.index) != DEFENSIVE]
    return {
        "policy": policy,
        "result": result,
        "analytics": analytics,
        "context": {
            "generated_at": report.generated_at(),
            "policy": policy,
            "shortlist_validated": validated,
            "accepted": result.accepted,
            "equity_share": result.equity_share,
            "metrics": analytics.metrics,
            "equity_only": analytics.equity_only_metrics,
            "worst_stress": float(analytics.stress["max_drawdown"].max()),
            "mc": result.details["monte_carlo"],
            "mc_history": result.details["mc_history"],
            "weights": analytics.weights,
            "checks": analytics.checks,
            "trials": analytics.trials,
            "foreign_weight": risk.foreign_currency_weight(
                result.weights, meta["currency"], policy.base_currency),
            "effective_lines": risk.effective_number_of_lines(equity / equity.sum()),
            "data_notes": data_notes,
            "explanation": [],
        },
    }


def write_outputs(out: dict) -> None:
    OUTPUT.mkdir(exist_ok=True)
    ctx = dict(out["context"], charts=report.make_charts(out["analytics"], out["policy"]))
    (OUTPUT / "rapport.html").write_text(report.render_html(ctx), encoding="utf-8")
    report.write_excel(out["analytics"], OUTPUT / "allocation.xlsx")
    print("Rapport : output/rapport.html · Export : output/allocation.xlsx")


# %% Exécution
if __name__ == "__main__":
    out = run()
    write_outputs(out)
    failed = out["result"].report.failed
    if failed:
        print("Contrôles non respectés :", ", ".join(c.name for c in failed))
