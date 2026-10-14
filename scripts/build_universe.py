"""Construit config/universe.csv à partir de la composition d'indices publics.

L'univers vient d'indices (pas d'un choix manuel de titres) :
- Euro Stoxx 50 (zone euro), FTSE 100 (Royaume-Uni), SMI (Suisse), S&P 100 (US).
Source : tableaux Wikipedia, à la date d'exécution du script.
Pays, secteur et industrie viennent ensuite de Yahoo Finance.

Usage : python scripts/build_universe.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import get_fundamentals  # noqa: E402

UNIVERSE_PATH = ROOT / "config" / "universe.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-copilot; usage personnel)"}


def _read_table(url: str, required_columns: set[str]) -> pd.DataFrame:
    html = requests.get(url, headers=HEADERS, timeout=30).text
    for table in pd.read_html(io.StringIO(html)):
        if required_columns <= set(table.columns):
            return table
    raise RuntimeError(f"Tableau des composants introuvable sur {url}")


def sp100() -> pd.DataFrame:
    t = _read_table("https://en.wikipedia.org/wiki/S%26P_100", {"Symbol", "Name"})
    return pd.DataFrame({"ticker": t["Symbol"].str.replace(".", "-", regex=False),
                         "index": "S&P 100"})


def eurostoxx50() -> pd.DataFrame:
    t = _read_table("https://en.wikipedia.org/wiki/EURO_STOXX_50", {"Ticker", "Main listing"})
    return pd.DataFrame({"ticker": t["Ticker"], "index": "Euro Stoxx 50"})


def ftse100() -> pd.DataFrame:
    t = _read_table("https://en.wikipedia.org/wiki/FTSE_100_Index", {"Company", "Ticker"})
    codes = t["Ticker"].str.rstrip(".").str.replace(".", "-", regex=False)
    return pd.DataFrame({"ticker": codes + ".L", "index": "FTSE 100"})


def smi() -> pd.DataFrame:
    t = _read_table("https://en.wikipedia.org/wiki/Swiss_Market_Index", {"Ticker", "Canton"})
    return pd.DataFrame({"ticker": t["Ticker"].dropna() + ".SW", "index": "SMI"})


def build() -> pd.DataFrame:
    members = pd.concat([eurostoxx50(), ftse100(), smi(), sp100()], ignore_index=True)
    members["ticker"] = members["ticker"].str.strip()
    members = members.drop_duplicates("ticker")
    print(f"{len(members)} titres dans les indices, récupération des fondamentaux...")
    fundamentals = get_fundamentals(members["ticker"].tolist())
    universe = members.merge(
        fundamentals[["name", "country", "sector", "industry", "currency"]],
        left_on="ticker", right_index=True, how="left",
    )
    universe["name"] = universe["name"].str.split().str.join(" ")
    return universe[["ticker", "name", "index", "country", "sector", "industry", "currency"]]


if __name__ == "__main__":
    universe = build()
    missing = universe[universe["industry"].isna()]
    universe.to_csv(UNIVERSE_PATH, index=False)
    print(f"{len(universe)} titres écrits dans {UNIVERSE_PATH.relative_to(ROOT)}")
    if not missing.empty:
        print(f"{len(missing)} sans données Yahoo (à vérifier) :", ", ".join(missing["ticker"]))
