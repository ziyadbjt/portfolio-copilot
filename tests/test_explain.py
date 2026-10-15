"""L'explication ne contient que des chiffres issus des calculs, et suit les bonnes branches."""


from src import report
from src.explain import binding_constraints, explanation_facts, render_explanation
from src.optimizer import Trial, optimize
from tests.conftest import with_cash


def build(policy, market, validated=True):
    prices, meta = market
    pol = with_cash(policy)
    result = optimize(prices, meta, pol)
    a = report.build_analytics(result, prices, meta, meta.index.to_series(), pol)
    return result, a, pol, explanation_facts(result, a, pol, validated)


def test_binding_constraint_is_the_closest_refused_share(policy, market):
    result, *_ = build(policy, market)
    result.trials = [Trial(0.2, True, "acceptée"), Trial(0.5, False, "stress crise_2008"),
                     Trial(0.4, False, "stress covid_mars_2020, P(drawdown > 20%) sur 5 ans"),
                     Trial(0.38, True, "acceptée")]
    result.equity_share = 0.38
    assert binding_constraints(result) == ["stress covid_mars_2020", "P(drawdown > 20%) sur 5 ans"]


def test_explanation_quotes_the_computed_numbers(policy, market):
    result, a, pol, facts = build(policy, market)
    text = " ".join(render_explanation(facts))
    assert f"{result.equity_share:.0%}".replace(".", ",") in text
    assert f"{facts['worst_drawdown']:.1%}".replace(".", ",") in text
    assert f"{facts['mc'].breach_probability:.1%}".replace(".", ",") in text
    assert "respecte toutes les règles" in text
    assert "ne garantissent rien" in text
    assert "pas encore validée" not in text


def test_explanation_flags_unvalidated_shortlist(policy, market):
    *_, facts = build(policy, market, validated=False)
    assert "pas encore validée" in " ".join(render_explanation(facts))


def test_explanation_when_policy_cannot_be_met(policy, market):
    *_, facts = build(policy, market)
    facts = dict(facts, accepted=False, failed=["stress crise_2008"])
    text = " ".join(render_explanation(facts))
    assert "Aucune allocation ne respecte" in text
    assert "stress crise_2008" in text
    assert "ne contourne pas une règle" in text


def test_explanation_warns_when_monte_carlo_misses_2008(policy, market):
    *_, facts = build(policy, market)
    import pandas as pd
    facts = dict(facts, mc_misses_2008=True, mc_start=pd.Timestamp("2015-03-20"))
    assert "ne contient donc pas la crise de 2008" in " ".join(render_explanation(facts))


def test_no_empty_paragraph(policy, market):
    *_, facts = build(policy, market)
    assert all(p.strip() for p in render_explanation(facts))
