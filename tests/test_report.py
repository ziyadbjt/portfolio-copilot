"""Le rapport se construit de bout en bout sur le marché simulé, sans réseau."""

import numpy as np

from src import report
from src.optimizer import optimize
from tests.conftest import with_cash


def test_report_end_to_end(policy, market, tmp_path):
    prices, meta = market
    pol = with_cash(policy)
    result = optimize(prices, meta, pol)
    names = meta.index.to_series()
    a = report.build_analytics(result, prices, meta, names, pol)

    # Le Monte Carlo du rapport est bien celui des contrôles
    replay = (a.mc_max_drawdowns > pol.risk.max_drawdown).mean()
    assert np.isclose(replay, result.details["monte_carlo"].breach_probability)
    assert abs(a.pockets.sum() - 1) < 1e-9
    assert abs(a.risk_contrib.sum() - 1) < 1e-9

    charts = report.make_charts(a, pol)
    assert all(len(b64) > 1000 for b64 in charts.values())

    html = report.render_html({
        "generated_at": "test", "policy": pol, "shortlist_validated": True,
        "accepted": result.accepted, "equity_share": result.equity_share,
        "metrics": a.metrics, "equity_only": a.equity_only_metrics,
        "worst_stress": float(a.stress["max_drawdown"].max()),
        "mc": result.details["monte_carlo"], "mc_history": result.details["mc_history"],
        "weights": a.weights, "checks": a.checks, "trials": a.trials,
        "foreign_weight": 0.1, "effective_lines": 5.0, "data_notes": [],
        "explanation": ["Paragraphe de test."], "charts": charts,
    })
    assert "Allocation conforme" in html
    assert "Paragraphe de test." in html
    # Aucune valeur manquante affichée (hors images, dont le base64 peut contenir "nan")
    import re
    text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html)
    assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)

    report.write_excel(a, tmp_path / "allocation.xlsx")
    assert (tmp_path / "allocation.xlsx").stat().st_size > 0
