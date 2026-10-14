"""Lance le screening sur l'univers et écrit la short-list proposée.

Usage :
  python scripts/run_screening.py            -> propose (output/screening.xlsx + shortlist_proposee.csv)
  python scripts/run_screening.py --valider  -> copie la proposition relue dans config/shortlist.csv

La validation est un geste humain explicite : config/shortlist.csv est versionné,
donc git garde la trace de chaque short-list validée et de sa date.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import get_fundamentals, load_market_data  # noqa: E402
from src.policy import load_policy  # noqa: E402
from src.screener import screen  # noqa: E402

UNIVERSE_PATH = ROOT / "config" / "universe.csv"
VALIDATED_PATH = ROOT / "config" / "shortlist.csv"
OUTPUT_DIR = ROOT / "output"
PROPOSAL_PATH = OUTPUT_DIR / "shortlist_proposee.csv"

COLUMNS_FR = {
    "name": "Nom", "pocket": "Poche", "theme": "Thème", "country": "Pays",
    "sector": "Secteur", "industry": "Industrie", "status": "Statut", "rank": "Rang",
    "score": "Score", "dividend_yield": "Rendement 12m", "payout_ratio": "Payout",
    "dividend_growth": "Croissance dividende/an", "dividend_cuts": "Baisses du dividende",
    "special_dividends": "Dividendes exceptionnels ignorés",
    "volatility": "Volatilité annuelle", "market_cap": "Capitalisation", "reasons": "Raisons",
}


def propose() -> None:
    policy = load_policy()
    universe = pd.read_csv(UNIVERSE_PATH, index_col="ticker")
    tickers = universe.index.tolist()

    end = dt.date.today()
    start = end - dt.timedelta(days=365 * policy.screening.history_years + 7)
    print(f"Chargement de {len(tickers)} titres du {start} au {end}...")
    md = load_market_data(tickers, start, end, policy.base_currency)
    fundamentals = get_fundamentals(tickers)

    result = screen(universe, fundamentals, md.prices, md.dividends, md.dividends_local, policy)
    shortlist = result[result["status"] == "shortlist"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    columns = list(COLUMNS_FR)
    with pd.ExcelWriter(OUTPUT_DIR / "screening.xlsx") as xl:
        shortlist[columns].rename(columns=COLUMNS_FR).to_excel(xl, sheet_name="Short-list")
        result[columns].rename(columns=COLUMNS_FR).to_excel(xl, sheet_name="Tous les titres")
        if not md.suspicious.empty:
            md.suspicious.to_excel(xl, sheet_name="Anomalies de prix", index=False)
    shortlist[["name", "pocket", "theme", "country", "sector", "score"]].to_csv(PROPOSAL_PATH)

    print(f"\n{result['status'].value_counts().to_dict()}")
    print("Raisons d'exclusion les plus fréquentes :")
    reasons = result.loc[result["status"] == "ecarte", "reasons"].str.split("; ").explode()
    print(reasons.value_counts().head(6).to_string())
    if not md.suspicious.empty:
        print(f"\n{len(md.suspicious)} variations de prix suspectes (onglet Anomalies)")

    view = shortlist[["name", "pocket", "theme", "country", "sector", "dividend_yield",
                      "payout_ratio", "volatility", "score"]].copy()
    for c in ["dividend_yield", "payout_ratio", "volatility"]:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    view["score"] = view["score"].round(2)
    print(f"\nShort-list proposée ({len(shortlist)} titres) :\n{view.to_string()}")
    print(f"\nDétail : output/screening.xlsx. Après relecture : --valider")


def validate() -> None:
    if not PROPOSAL_PATH.exists():
        sys.exit("Aucune proposition : lancer d'abord le script sans --valider.")
    shutil.copy(PROPOSAL_PATH, VALIDATED_PATH)
    n = len(pd.read_csv(VALIDATED_PATH))
    print(f"Short-list validée ({n} titres) -> {VALIDATED_PATH.relative_to(ROOT)}")
    print("Pense à la committer pour garder la trace de la décision.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valider", action="store_true", help="valider la proposition relue")
    args = parser.parse_args()
    validate() if args.valider else propose()
