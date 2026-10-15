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
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# Devises cotées en centièmes : (devise réelle, diviseur)
MINOR_CURRENCIES = {"GBp": ("GBP", 100), "GBX": ("GBP", 100)}


class DataError(RuntimeError):
    """Données introuvables ou inutilisables."""


# --- Accès réseau (isolé) ----------------------------------------------------

def _with_retries(fn, what: str, attempts: int = 3, wait_s: float = 2.0):
    """Réessaie un appel réseau ; au dernier échec, lève une DataError lisible."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except DataError:
            raise                                # réponse claire de Yahoo : inutile de réessayer
        except Exception as e:                   # coupure, délai dépassé, limite de requêtes...
            if attempt == attempts:
                raise DataError(f"{what} : échec après {attempts} essais ({type(e).__name__}).") from e
            time.sleep(wait_s * attempt)


def _download_history(ticker: str, start: dt.date, end: dt.date) -> tuple[pd.DataFrame, str]:
    """Télécharge prix ajustés et dividendes. Renvoie (DataFrame[close, dividends], devise)."""
    import yfinance as yf

    # Sans ça, yfinance masque les erreurs réseau et renvoie un tableau vide :
    # un simple délai dépassé passerait pour "pas de données", sans réessai
    yf.config.debug.hide_exceptions = False
    tk = yf.Ticker(ticker)
    # yfinance traite end comme exclusif : on ajoute un jour
    hist = _with_retries(
        lambda: tk.history(start=start, end=end + dt.timedelta(days=1), auto_adjust=True),
        f"Historique {ticker}",
    )
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


FUNDAMENTAL_FIELDS = {          # nom Yahoo -> nom dans le projet
    "shortName": "name",
    "country": "country",
    "sector": "sector",
    "industry": "industry",
    "payoutRatio": "payout_ratio",
    "marketCap": "market_cap",
    "currency": "currency",
}


def _download_fundamentals(ticker: str) -> dict:
    import yfinance as yf

    info = _with_retries(lambda: yf.Ticker(ticker).info, f"Fondamentaux {ticker}") or {}
    return {ours: info.get(theirs) for theirs, ours in FUNDAMENTAL_FIELDS.items()}


def get_fundamentals(
    tickers: list[str], cache_dir: Path = DEFAULT_CACHE_DIR, max_age_days: int = 7
) -> pd.DataFrame:
    """Pays, secteur, industrie, payout... par ticker (une ligne par ticker).

    Le rendement du dividende n'est pas pris ici : celui de Yahoo est faux pour
    les titres cotés en pence. On le recalcule à partir de l'historique.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "fundamentals.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    today = dt.date.today()

    rows = {}
    for ticker in tickers:
        entry = cache.get(ticker)
        if entry is None or (today - dt.date.fromisoformat(entry["fetched"])).days >= max_age_days:
            try:
                entry = {**_download_fundamentals(ticker), "fetched": today.isoformat()}
            except Exception as e:  # yfinance lève des erreurs variées (HTTP, JSON...)
                entry = {"error": str(e), "fetched": today.isoformat()}
            cache[ticker] = entry
            path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        rows[ticker] = {k: entry.get(k) for k in FUNDAMENTAL_FIELDS.values()}

    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("ticker")


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


def fill_before_listing(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Prix constant avant la première cotation de `ticker` (rendement nul).

    Réservé au fonds monétaire de la poche défensive : avant son lancement,
    on suppose qu'il ne rapporte rien. Hypothèse prudente (le monétaire rapportait
    ~4 %/an en 2007) qui évite d'exclure la poche défensive du stress test 2008.
    Ne jamais l'utiliser pour une action : ce serait inventer un historique.
    """
    out = prices.copy()
    out[ticker] = out[ticker].bfill()
    return out


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
    dividends_local: pd.DataFrame   # dividendes par action, en devise de cotation (sans effet change)
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
    min_coverage: float = 0.9,
) -> MarketData:
    """Prix et dividendes de `tickers` en devise de base, prêts pour le calcul de risque.

    min_coverage : part minimale de la période couverte par un titre. Pour un
    stress test sur 2008, mettre 0 : un titre coté plus tard est gardé et le
    stress test indique la part du portefeuille qu'il couvre vraiment.
    """
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

    prices, cleaning_dropped = clean_prices(prices, min_coverage=min_coverage)
    dropped.update(cleaning_dropped)
    dividends = dividends.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    divs_local = divs_local.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    return MarketData(
        prices=prices,
        dividends=dividends,
        dividends_local=divs_local,
        currencies={t: c for t, c in currencies.items() if t in prices.columns},
        fx=fx,
        dropped=dropped,
        suspicious=find_suspicious_jumps(prices),
    )
