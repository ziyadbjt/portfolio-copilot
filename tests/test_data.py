import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import data
from src.data import (
    DataError, clean_prices, find_suspicious_jumps, fx_ticker,
    get_history, load_market_data, normalize_currency, to_base_currency,
)


def days(*isodates):
    return pd.DatetimeIndex(pd.to_datetime(isodates))


# --- Fonctions pures ---------------------------------------------------------

def test_pence_converted_to_pounds():
    s, ccy = normalize_currency(pd.Series([4274.0]), "GBp")
    assert ccy == "GBP"
    assert s.iloc[0] == pytest.approx(42.74)


def test_other_currency_unchanged():
    s, ccy = normalize_currency(pd.Series([100.0]), "USD")
    assert ccy == "USD" and s.iloc[0] == 100.0


def test_fx_ticker():
    assert fx_ticker("USD", "EUR") == "USDEUR=X"


def test_to_base_currency_known_values():
    idx = days("2024-01-02", "2024-01-03")
    prices = pd.DataFrame({"MSFT": [100.0, 110.0], "TTE.PA": [60.0, 61.0]}, index=idx)
    fx = {"USD": pd.Series([0.90, 0.95], index=idx)}
    out = to_base_currency(prices, {"MSFT": "USD", "TTE.PA": "EUR"}, fx, "EUR")
    assert out["MSFT"].tolist() == pytest.approx([90.0, 104.5])
    assert out["TTE.PA"].tolist() == [60.0, 61.0]


def test_to_base_currency_uses_last_known_rate():
    # Le marché des changes n'a pas de cotation le 3 : on reprend le taux du 2
    prices = pd.DataFrame({"MSFT": [100.0, 100.0]}, index=days("2024-01-02", "2024-01-03"))
    fx = {"USD": pd.Series([0.90], index=days("2024-01-02"))}
    out = to_base_currency(prices, {"MSFT": "USD"}, fx, "EUR")
    assert out["MSFT"].tolist() == pytest.approx([90.0, 90.0])


def test_to_base_currency_missing_rate():
    prices = pd.DataFrame({"X": [1.0]}, index=days("2024-01-02"))
    with pytest.raises(DataError, match="CHF"):
        to_base_currency(prices, {"X": "CHF"}, {}, "EUR")


def test_clean_fills_holiday_but_not_before_listing():
    idx = pd.bdate_range("2024-01-01", periods=20)
    paris = pd.Series(np.arange(20, dtype=float) + 50, index=idx)
    paris.iloc[5] = np.nan                       # jour férié à Paris
    ipo = pd.Series(np.nan, index=idx)
    ipo.iloc[1:] = 10.0                          # coté à partir du 2e jour (95 % de couverture)
    clean, dropped = clean_prices(pd.DataFrame({"PAR": paris, "IPO": ipo}))
    assert clean["PAR"].iloc[5] == clean["PAR"].iloc[4]   # prix reporté
    assert np.isnan(clean["IPO"].iloc[0])                 # pas d'historique inventé
    assert dropped == {}


def test_clean_drops_short_history():
    idx = pd.bdate_range("2024-01-01", periods=20)
    short = pd.Series(np.nan, index=idx)
    short.iloc[-5:] = 1.0                        # 25 % de couverture
    clean, dropped = clean_prices(pd.DataFrame({"OK": 1.0, "SHORT": short}, index=idx))
    assert list(clean.columns) == ["OK"]
    assert "SHORT" in dropped


def test_clean_does_not_fill_long_gaps():
    idx = pd.bdate_range("2024-01-01", periods=30)
    s = pd.Series(1.0, index=idx)
    s.iloc[10:20] = np.nan                       # 10 jours de trou > max_gap=5
    # REF cote tous les jours : sinon les lignes vides seraient vues comme des fériés communs
    frame = pd.DataFrame({"X": s, "REF": 1.0})
    clean, _ = clean_prices(frame, max_gap=5, min_coverage=0.5)
    assert clean["X"].iloc[10:15].notna().all()
    assert clean["X"].iloc[15:20].isna().all()


def test_suspicious_jump_detected():
    prices = pd.DataFrame({"OK": [100.0, 101.0, 102.0], "BUG": [40.0, 4000.0, 40.0]},
                          index=days("2024-01-02", "2024-01-03", "2024-01-04"))
    jumps = find_suspicious_jumps(prices)
    assert set(jumps["ticker"]) == {"BUG"}
    assert len(jumps) == 2


# --- Cache et assemblage (réseau simulé) -------------------------------------

def fake_market(calls):
    """Faux _download_history : séries connues, et on compte les appels."""
    def _fake(ticker, start, end):
        calls.append(ticker)
        idx = pd.bdate_range(start, end)
        if ticker == "UNKNOWN":
            raise DataError("Aucune donnée Yahoo pour UNKNOWN.")
        table = {
            "TTE.PA": (60.0, "EUR"),
            "MSFT": (400.0, "USD"),
            "ULVR.L": (4000.0, "GBp"),
            "USDEUR=X": (0.9, "EUR"),
            "GBPEUR=X": (1.2, "EUR"),
        }
        price, ccy = table[ticker]
        divs = pd.Series(0.0, index=idx)
        divs.iloc[2] = price / 100
        return pd.DataFrame({"close": price, "dividends": divs}, index=idx), ccy
    return _fake


def test_cache_avoids_second_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "_download_history", fake_market(calls))
    start, end = dt.date(2020, 1, 1), dt.date(2020, 6, 30)
    first, _ = get_history("TTE.PA", start, end, tmp_path)
    second, ccy = get_history("TTE.PA", start, dt.date(2020, 3, 31), tmp_path)
    assert calls == ["TTE.PA"]
    assert ccy == "EUR"
    assert second.index.max() <= pd.Timestamp("2020-03-31")


def test_cache_extends_without_shrinking(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "_download_history", fake_market(calls))
    get_history("TTE.PA", dt.date(2020, 1, 1), dt.date(2020, 6, 30), tmp_path)
    get_history("TTE.PA", dt.date(2019, 1, 1), dt.date(2019, 6, 30), tmp_path)
    get_history("TTE.PA", dt.date(2019, 1, 1), dt.date(2020, 6, 30), tmp_path)
    assert calls == ["TTE.PA", "TTE.PA"]           # le 3e appel est servi par le cache


def test_load_market_data_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_download_history", fake_market([]))
    md = load_market_data(["TTE.PA", "MSFT", "ULVR.L", "UNKNOWN"],
                          dt.date(2020, 1, 1), dt.date(2020, 3, 31), "EUR", tmp_path)
    assert set(md.prices.columns) == {"TTE.PA", "MSFT", "ULVR.L"}
    assert "UNKNOWN" in md.dropped
    assert md.prices["TTE.PA"].iloc[0] == pytest.approx(60.0)
    assert md.prices["MSFT"].iloc[0] == pytest.approx(400 * 0.9)
    assert md.prices["ULVR.L"].iloc[0] == pytest.approx(40 * 1.2)   # pence -> GBP -> EUR
    assert md.dividends["ULVR.L"].sum() == pytest.approx(0.40 * 1.2)
    assert md.currencies == {"TTE.PA": "EUR", "MSFT": "USD", "ULVR.L": "GBP"}
    assert md.suspicious.empty


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("réseau")
        return "ok"

    assert data._with_retries(flaky, "test") == "ok"
    assert len(calls) == 3


def test_retries_exhausted_becomes_data_error(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda s: None)

    def always_down():
        raise TimeoutError("réseau")

    with pytest.raises(DataError, match="3 essais"):
        data._with_retries(always_down, "Historique X")
