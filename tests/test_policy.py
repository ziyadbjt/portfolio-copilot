import copy
import datetime as dt

import pytest
import yaml

from src.policy import DEFAULT_POLICY_PATH, PolicyError, load_policy, parse_policy


@pytest.fixture
def raw():
    with DEFAULT_POLICY_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_real_policy_loads():
    p = load_policy()
    assert p.base_currency == "EUR"
    assert p.horizon_years == 5
    assert p.risk.max_drawdown == 0.20
    assert p.allocation.core_target == 0.80
    assert p.risk.mc_breach_probability_max == 0.10
    assert {s.name for s in p.risk.stress_scenarios} == {
        "crise_2008", "covid_mars_2020", "inflation_2022"
    }
    assert "Tobacco" in p.excluded_industries


def test_policy_is_immutable():
    p = load_policy()
    with pytest.raises(Exception):
        p.risk.max_drawdown = 0.5


def test_missing_file():
    with pytest.raises(PolicyError, match="introuvable"):
        load_policy("nexiste/pas.yaml")


def test_missing_field(raw):
    del raw["risk"]["max_drawdown"]
    with pytest.raises(PolicyError, match="max_drawdown"):
        parse_policy(raw)


@pytest.mark.parametrize("bad", [1.5, -0.1, "20%", True])
def test_fraction_out_of_range_or_wrong_type(raw, bad):
    raw["limits"]["max_line_weight"] = bad
    with pytest.raises(PolicyError, match="max_line_weight"):
        parse_policy(raw)


def test_core_plus_satellite_must_sum_to_one(raw):
    raw["allocation"]["core_target"] = 0.70
    with pytest.raises(PolicyError, match="doit faire 1"):
        parse_policy(raw)


def test_satellite_target_above_max(raw):
    raw["allocation"].update(core_target=0.70, satellite_target=0.30, satellite_max=0.25)
    with pytest.raises(PolicyError, match="satellite_max"):
        parse_policy(raw)


def test_line_above_sector(raw):
    raw["limits"]["max_line_weight"] = 0.40
    with pytest.raises(PolicyError, match="max_sector_weight"):
        parse_policy(raw)


def test_bad_scenario_date(raw):
    raw["risk"]["stress_scenarios"]["inflation_2022"]["start"] = "2022-0103"
    with pytest.raises(PolicyError, match="inflation_2022"):
        parse_policy(raw)


def test_scenario_dates_inverted(raw):
    raw["risk"]["stress_scenarios"]["covid_mars_2020"] = {
        "start": dt.date(2020, 3, 23), "end": dt.date(2020, 2, 19)
    }
    with pytest.raises(PolicyError, match="précéder"):
        parse_policy(raw)


def test_parse_does_not_mutate_input(raw):
    before = copy.deepcopy(raw)
    parse_policy(raw)
    assert raw == before
