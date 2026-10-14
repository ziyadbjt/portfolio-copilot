"""Récupération et nettoyage des prix et dividendes.

C'est le seul module qui parle à yfinance : si la source change, seul ce fichier
change. Le reste du projet reçoit des DataFrames propres, en devise de base.

Choix :
- prix "ajustés" (auto_adjust=True) : dividendes et splits réinvestis, donc les
  rendements calculés dessus sont des rendements totaux ;
- cache local en parquet par ticker, pour ne pas solliciter Yahoo à chaque run ;
- les fonctions de nettoyage et de conversion sont pures (sans réseau), donc testables.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# Devises cotées en centièmes : (devise réelle, diviseur)
MINOR_CURRENCIES = {"GBp": ("GBP", 100), "GBX": ("GBP", 100)}


class DataError(RuntimeError):
    """Données introuvables ou inutilisables."""


# --- Accès réseau (isolé) ----------------------------------------------------

def _download_history(ticker: str, start: dt.date, end: dt.date) -> tuple[pd.DataFrame, str]:
    """Télécharge prix ajustés et dividendes. Renvoie (DataFrame[close, dividends], devise)."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    # yfinance traite end comme exclusif : on ajoute un jour
    hist = tk.history(start=start, end=end + dt.timedelta(days=1), auto_adjust=True)
    if hist.empty:
        raise DataError(f"Aucune donnée Yahoo pour {ticker}.")
    currency = tk.history_metadata.get("currency")
    if not currency:
        raise DataError(f"Devise inconnue pour {ticker}.")
    df = pd.DataFrame({"close": hist["Close"], "dividends": hist["Dividends"]})
    # Dates sans fuseau : Paris et New York doivent partager le même index
    df.index = pd.DatetimeIndex(df.index.tz_localize(None).normalize(), name="date")
    return df, currency


# --- Cache -------------------------------------------------------------------

def _cache_paths(cache_dir: Path, ticker: str) -> tuple[Path, Path]:
    safe = ticker.replace("=", "_").replace("/", "_").replace("^", "_")
    return cache_dir / f"{safe}.parquet", cache_dir / "index.json"


def _read_index(index_path: Path) -> dict:
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return {}


def get_history(
    ticker: str,
    start: dt.date,
    end: dt.date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_age_days: int = 1,
) -> tuple[pd.DataFrame, str]:
    """Historique d'un ticker (close, dividends) et sa devise, via le cache si possible.

    Le cache est réutilisé s'il couvre [start, end] et, quand la période touche
    les derniers jours, s'il a moins de max_age_days.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, index_path = _cache_paths(cache_dir, ticker)
    index = _read_index(index_path)
    meta = index.get(ticker)
    today = dt.date.today()

    if meta and data_path.exists():
        covers = (dt.date.fromisoformat(meta["start"]) <= start
                  and dt.date.fromisoformat(meta["end"]) >= end)
        fresh = end < today - dt.timedelta(days=max_age_days) or \
            (today - dt.date.fromisoformat(meta["fetched"])).days < max_age_days
        if covers and fresh:
            df = pd.read_parquet(data_path)
            return df.loc[pd.Timestamp(start):pd.Timestamp(end)], meta["currency"]

    # On retélécharge l'union des périodes pour ne pas rétrécir le cache existant
    fetch_start, fetch_end = start, end
    if meta:
        fetch_start = min(start, dt.date.fromisoformat(meta["start"]))
        fetch_end = max(end, dt.date.fromisoformat(meta["end"]))
    df, currency = _download_history(ticker, fetch_start, fetch_end)
    df.to_parquet(data_path)
    index[ticker] = {"start": fetch_start.isoformat(), "end": fetch_end.isoformat(),
                     "fetched": today.isoformat(), "currency": currency}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)], currency


# --- Fonctions pures ---------------------------------------------------------

def normalize_currency(prices: pd.Series, currency: str) -> tuple[pd.Series, str]:
    """Convertit une cotation en centièmes (GBp) vers l'unité (GBP)."""
    if currency in MINOR_CURRENCIES:
        real, divisor = MINOR_CURRENCIES[currency]
        return prices / divisor, real
    return prices, currency


def fx_ticker(currency: str, base: str) -> str:
    """Ticker Yahoo donnant le nombre d'unités de `base` pour 1 unité de `currency`."""
    return f"{currency}{base}=X"


def to_base_currency(
    prices: pd.DataFrame, currencies: dict[str, str], fx: dict[str, pd.Series], base: str
) -> pd.DataFrame:
    """Convertit chaque colonne de prix en devise de base.

    fx[devise] = combien de `base` vaut 1 unité de devise, à chaque date.
    Un taux manquant un jour donné est remplacé par le dernier connu.
    """
    out = {}
    for ticker in prices.columns:
        ccy = currencies[ticker]
        if ccy == base:
            out[ticker] = prices[ticker]
            continue
        if ccy not in fx:
            raise DataError(f"Taux de change {ccy}->{base} manquant pour {ticker}.")
        rate = fx[ccy].reindex(prices.index.union(fx[ccy].index)).sort_index().ffill()
        out[ticker] = prices[ticker] * rate.reindex(prices.index)
    return pd.DataFrame(out, index=prices.index)


def clean_prices(
    prices: pd.DataFrame, max_gap: int = 5, min_coverage: float = 0.9
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Aligne les places de cotation et retire les séries inutilisables.

    - jours fériés propres à une place : prix reporté (rendement nul ce jour-là),
      au plus `max_gap` jours de suite pour ne pas masquer une vraie absence ;
    - aucun remplissage avant la première cotation (pas d'historique inventé) ;
    - un ticker couvrant moins de `min_coverage` de la période est écarté.

    Renvoie (prix nettoyés, {ticker écarté: raison}).
    """
    prices = prices.sort_index().dropna(how="all")
    filled = prices.ffill(limit=max_gap)
    dropped = {}
    for ticker in filled.columns:
        coverage = filled[ticker].notna().mean()
        if coverage < min_coverage:
            dropped[ticker] = f"couverture {coverage:.0%} < {min_coverage:.0%}"
    return filled.drop(columns=list(dropped)), dropped


def find_suspicious_jumps(prices: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Repère les variations journalières anormales (souvent des erreurs Yahoo).

    Signale sans corriger : c'est à l'humain de trancher.
    """
    returns = prices.pct_change(fill_method=None).stack()
    jumps = returns[returns.abs() > threshold]
    return jumps.rename_axis(["date", "ticker"]).rename("return").reset_index()


# --- Point d'entrée ----------------------------------------------------------

@dataclass
class MarketData:
    prices: pd.DataFrame            # prix ajustés en devise de base, nettoyés
    dividends: pd.DataFrame         # dividendes par action, en devise de base
    currencies: dict[str, str]      # devise de cotation de chaque ticker (après GBp -> GBP)
    fx: dict[str, pd.Series]        # taux utilisés (base par unité de devise)
    dropped: dict[str, str] = field(default_factory=dict)
    suspicious: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_market_data(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
    base: str = "EUR",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> MarketData:
    """Prix et dividendes de `tickers` en devise de base, prêts pour le calcul de risque."""
    closes, divs, currencies, dropped = {}, {}, {}, {}
    for ticker in tickers:
        try:
            hist, ccy = get_history(ticker, start, end, cache_dir)
        except DataError as e:
            dropped[ticker] = str(e)
            continue
        closes[ticker], currencies[ticker] = normalize_currency(hist["close"], ccy)
        divs[ticker], _ = normalize_currency(hist["dividends"], ccy)

    if not closes:
        raise DataError("Aucun ticker n'a pu être chargé.")

    fx = {}
    for ccy in sorted(set(currencies.values()) - {base}):
        hist, _ = get_history(fx_ticker(ccy, base), start, end, cache_dir)
        fx[ccy] = hist["close"]

    prices_local = pd.DataFrame(closes)
    divs_local = pd.DataFrame(divs).reindex(prices_local.index).fillna(0.0)
    prices = to_base_currency(prices_local, currencies, fx, base)
    dividends = to_base_currency(divs_local, currencies, fx, base)

    prices, cleaning_dropped = clean_prices(prices)
    dropped.update(cleaning_dropped)
    dividends = dividends.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    return MarketData(
        prices=prices,
        dividends=dividends,
        currencies={t: c for t, c in currencies.items() if t in prices.columns},
        fx=fx,
        dropped=dropped,
        suspicious=find_suspicious_jumps(prices),
    )
