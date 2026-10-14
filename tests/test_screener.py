import numpy as np
import pandas as pd
import pytest

from src.policy import load_policy
from src.screener import (
    annual_dividends, annualized_volatility, assign_pocket, count_dividend_cuts,
    dividend_growth, percentile_score, screen, select_shortlist, special_dividend_mask,
    trailing_dividend_yield,
)

# --- Mesures sur des cas dont on connaît le résultat -------------------------


def test_assign_pocket():
    themes = {"IA": ("Semiconductors",), "Energie": ("Oil & Gas Integrated",)}
    industry = pd.Series({"MU": "Semiconductors", "TTE.PA": "Oil & Gas Integrated",
                          "NESN.SW": "Packaged Foods"})
    p = assign_pocket(industry, themes)
    assert p.loc["MU"].tolist() == ["satellite", "IA"]
    assert p.loc["TTE.PA"].tolist() == ["satellite", "Energie"]
    assert p.loc["NESN.SW", "pocket"] == "coeur" and pd.isna(p.loc["NESN.SW", "theme"])


def test_trailing_yield_counts_last_12_months_only():
    idx = pd.to_datetime(["2023-06-01", "2024-03-01", "2024-09-01", "2024-12-31"])
    divs = pd.DataFrame({"X": [9.0, 1.0, 1.0, 0.0]}, index=idx)   # 9 € il y a plus d'un an
    prices = pd.DataFrame({"X": [100.0, 100.0, 100.0, 50.0]}, index=idx)
    y = trailing_dividend_yield(divs, prices, pd.Timestamp("2024-12-31"))
    assert y["X"] == pytest.approx(2.0 / 50.0)                   # dernier prix, pas la moyenne


def test_annual_dividends_fills_missing_years_with_zero():
    idx = pd.to_datetime(["2021-05-01", "2021-11-01", "2023-05-01"])
    divs = pd.DataFrame({"X": [1.0, 1.0, 2.5]}, index=idx)
    a = annual_dividends(divs, last_year=2023, n_years=3)
    assert a["X"].tolist() == [2.0, 0.0, 2.5]


def test_count_cuts_with_tolerance():
    annual = pd.DataFrame({"STABLE": [1.0, 1.0, 1.1, 1.2],
                           "PETITE_BAISSE": [1.0, 0.95, 1.0, 1.0],   # -5 % : toléré
                           "COUPE": [1.0, 0.5, 0.6, 0.3]})            # 2 vraies coupes
    cuts = count_dividend_cuts(annual, tolerance=0.10)
    assert cuts.to_dict() == {"STABLE": 0, "PETITE_BAISSE": 0, "COUPE": 2}


def test_dividend_growth_cagr():
    annual = pd.DataFrame({"X": [1.0, 1.1, 1.21], "NEW": [0.0, 1.0, 1.0]})
    g = dividend_growth(annual)
    assert g["X"] == pytest.approx(0.10)
    assert np.isnan(g["NEW"])          # pas de croissance calculable depuis zéro


def test_volatility_of_known_series():
    # Rendements alternés +1 % / -1 % : écart-type journalier connu
    idx = pd.bdate_range("2020-01-01", periods=1001)
    r = np.where(np.arange(1000) % 2 == 0, 0.01, -0.01)
    prices = pd.DataFrame({"X": 100 * np.cumprod(np.r_[1, 1 + r])}, index=idx)
    vol = annualized_volatility(prices, years=10, asof=idx[-1])
    expected = pd.Series(r).std() * np.sqrt(252)
    assert vol["X"] == pytest.approx(expected)


def test_percentile_score_direction():
    m = pd.DataFrame({"dividend_yield": [0.05, 0.03, 0.01],
                      "volatility": [0.10, 0.20, 0.30]}, index=["BEST", "MID", "WORST"])
    s = percentile_score(m, {"dividend_yield": 0.5, "volatility": 0.5})
    assert s["BEST"] > s["MID"] > s["WORST"]
    assert s["BEST"] == pytest.approx(1.0)


def test_shortlist_respects_sector_cap():
    c = pd.DataFrame({"score": [0.9, 0.8, 0.7, 0.6],
                      "sector": ["Banks", "Banks", "Banks", "Utilities"]},
                     index=["B1", "B2", "B3", "U1"])
    assert select_shortlist(c, size=3, max_per_sector=2) == ["B1", "B2", "U1"]


# --- Screening complet sur un mini-univers simulé ----------------------------

@pytest.fixture
def mini_market():
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    rng = np.random.default_rng(0)
    tickers = ["GOOD", "HIGHPAY", "CUTTER", "SMOKE", "CHIP", "NODATA"]
    vol = {"GOOD": 0.01, "HIGHPAY": 0.01, "CUTTER": 0.01, "SMOKE": 0.01, "CHIP": 0.02}
    prices = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, v, len(idx))))
                           for t, v in vol.items()}, index=idx)
    divs = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    for year in range(2019, 2026):
        day = pd.Timestamp(f"{year}-06-03")
        divs.loc[day, ["GOOD", "HIGHPAY", "SMOKE"]] = 4.0
        divs.loc[day, "CUTTER"] = 4.0 if year % 2 else 1.0     # coupe une année sur deux
    universe = pd.DataFrame({
        "name": tickers,
        "country": "France",
        "sector": ["Utilities", "Utilities", "Industrials", "Consumer Defensive",
                   "Technology", "Utilities"],
        "industry": ["Utilities - Regulated Electric", "Utilities - Regulated Electric",
                     "Conglomerates", "Tobacco", "Semiconductors", "Utilities - Diversified"],
    }, index=pd.Index(tickers, name="ticker"))
    fundamentals = pd.DataFrame({
        "payout_ratio": [0.5, 0.95, 0.5, 0.6, 0.1, 0.5],
        "market_cap": [1e10] * 6,
    }, index=tickers)
    return universe, fundamentals, prices, divs


def test_screen_end_to_end(mini_market):
    universe, fundamentals, prices, divs = mini_market
    out = screen(universe, fundamentals, prices, divs, divs, load_policy())
    assert out.loc["GOOD", "status"] == "shortlist"
    assert out.loc["CHIP", "status"] == "shortlist"           # satellite, pas de filtre dividende
    assert out.loc["CHIP", "theme"] == "IA"
    assert "payout > 80%" in out.loc["HIGHPAY", "reasons"]
    assert "baisse(s) du dividende" in out.loc["CUTTER", "reasons"]
    assert out.loc["SMOKE", "reasons"] == "industrie exclue"
    assert out.loc["NODATA", "reasons"] == "historique de prix insuffisant"
    assert set(out.loc[out["status"] == "ecarte"].index) == {"HIGHPAY", "CUTTER", "SMOKE", "NODATA"}
    assert out["status"].tolist()[0] == "shortlist"          # short-list en tête


def test_special_dividend_detected_and_ignored():
    idx = pd.to_datetime(["2023-04-01", "2023-08-01", "2024-04-01", "2024-08-01",
                          "2025-02-01", "2025-04-01", "2025-08-01"])
    divs = pd.DataFrame({"RKT": [1.1, 0.8, 1.2, 0.8, 2.5, 1.2, 0.8]}, index=idx)
    mask = special_dividend_mask(divs, multiple=1.5)
    assert mask["RKT"].tolist() == [False] * 4 + [True, False, False]   # 2.5 = exceptionnel


def test_costco_style_special_with_quarterly_dividends():
    idx = pd.date_range("2022-01-15", periods=12, freq="QS-JAN")
    amounts = [0.9] * 8 + [15.0] + [1.0] * 3                 # 15 $ au milieu des trimestriels
    divs = pd.DataFrame({"COST": amounts}, index=idx)
    mask = special_dividend_mask(divs, multiple=1.5)["COST"]
    assert mask.tolist() == [False] * 8 + [True] + [False] * 3


def test_too_little_history_is_not_judged():
    # Données commençant juste avant l'acompte : le solde suivant n'a pas de comparaison valable
    idx = pd.to_datetime(["2020-10-01", "2020-11-01", "2021-06-01", "2021-11-01", "2022-06-01"])
    divs = pd.DataFrame({"UU": [0.0, 0.14, 0.29, 0.15, 0.30]}, index=idx)
    assert not special_dividend_mask(divs, multiple=1.5)["UU"].any()


def test_interim_and_final_are_not_special():
    idx = pd.to_datetime(["2024-05-01", "2024-09-01", "2025-05-01", "2025-09-01"])
    divs = pd.DataFrame({"BNP": [4.6, 2.6, 4.8, 2.6]}, index=idx)
    assert not special_dividend_mask(divs, multiple=1.5)["BNP"].any()


def test_small_interim_big_final_is_not_special():
    # ING : acompte ~5x plus petit que le solde, la médiane serait trompeuse
    idx = pd.to_datetime(["2023-05-01", "2023-08-01", "2024-05-01", "2024-08-01", "2025-05-01"])
    divs = pd.DataFrame({"ING": [0.75, 0.17, 0.80, 0.18, 0.85]}, index=idx)
    assert not special_dividend_mask(divs, multiple=1.5)["ING"].any()


def test_growing_dividend_is_not_special():
    idx = pd.to_datetime(["2021-05-01", "2022-05-01", "2023-05-01", "2024-05-01", "2025-05-01"])
    divs = pd.DataFrame({"DIS": [0.2, 0.3, 0.45, 0.5, 0.75]}, index=idx)  # +50 % durables
    assert not special_dividend_mask(divs, multiple=1.5)["DIS"].any()
