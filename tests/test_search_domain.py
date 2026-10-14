"""Generation is bounded, reproducible and independent of global randomness."""

import random
from dataclasses import FrozenInstanceError, replace
from random import Random

import pytest

from trialgrid.domain import BatteryConfig
from trialgrid.search.domain import (
    ScenarioDomain,
    mutate_scenario,
    sample_scenario,
    scenario_fingerprint,
    validate_scenario,
)


def flatten(scenario):
    return (scenario.initial_state.energy_kwh, *scenario.prices_eur_per_mwh,
            *(v for row in scenario.forecast_prices_eur_per_mwh for v in row),
            *scenario.reserve_requirements_kwh)


def test_sampling_repeatable_bounded_and_global_rng_untouched():
    domain = ScenarioDomain()
    before = random.getstate()
    scenario = sample_scenario(domain, Random(42))
    assert scenario == sample_scenario(domain, Random(42))
    assert random.getstate() == before
    validate_scenario(scenario, domain)
    assert len(scenario.prices_eur_per_mwh) == 24
    assert [len(row) for row in scenario.forecast_prices_eur_per_mwh][-4:] == [4, 3, 2, 1]
    assert domain.config.energy_floor_kwh <= scenario.initial_state.energy_kwh <= 90
    assert 0 <= scenario.seed < 2**64
    assert scenario.forecast_prices_eur_per_mwh[0] != scenario.prices_eur_per_mwh[:4]


def test_mutations_change_exactly_one_scalar_and_preserve_domain():
    domain = ScenarioDomain(hours=3)
    parent = sample_scenario(domain, Random(10))
    before = random.getstate()
    changed = set()
    for seed in range(100):
        result = mutate_scenario(parent, domain, Random(seed))
        assert result == mutate_scenario(parent, domain, Random(seed))
        validate_scenario(result, domain)
        differences = [
            i for i, (a, b) in enumerate(zip(flatten(parent), flatten(result), strict=True))
            if a != b
        ]
        assert len(differences) == 1
        changed.update(differences)
    assert len(changed) == len(flatten(parent))
    assert random.getstate() == before


def test_degenerate_ranges_mutate_only_initial_energy():
    domain = ScenarioDomain(price_min=0, price_max=0, reserve_min_kwh=2, reserve_max_kwh=2)
    parent = sample_scenario(domain, Random(1))
    child = mutate_scenario(parent, domain, Random(2))
    assert child.initial_state != parent.initial_state
    assert flatten(child)[1:] == flatten(parent)[1:]


def test_domain_strict_serialization_and_immutability():
    domain = ScenarioDomain()
    assert ScenarioDomain.from_json(domain.to_json()) == domain
    assert ScenarioDomain.from_dict(domain.to_dict()) == domain
    with pytest.raises(FrozenInstanceError):
        domain.hours = 2
    data = domain.to_dict()
    data['extra'] = 1
    with pytest.raises(ValueError):
        ScenarioDomain.from_dict(data)
    with pytest.raises(ValueError):
        ScenarioDomain.from_json('{"hours": 2, "hours": 3}')
    data = domain.to_dict()
    del data['config']['dt_h']
    with pytest.raises(ValueError):
        ScenarioDomain.from_dict(data)


@pytest.mark.parametrize('settings', [
    {'hours': 0}, {'hours': 169}, {'hours': True}, {'hours': 2.0},
    {'forecast_horizon': 0}, {'forecast_horizon': 7}, {'forecast_horizon': False},
    {'price_min': 4, 'price_max': 3}, {'price_min': float('nan')},
    {'price_max': float('inf')}, {'reserve_min_kwh': -1}, {'reserve_max_kwh': 101},
    {'reserve_min_kwh': 91}, {'reserve_max_kwh': True},
    {'config': BatteryConfig(100, 10, 90, 20, 20, dt_h=0.5)}, {'config': {}},
])
def test_invalid_domains(settings):
    with pytest.raises(ValueError):
        ScenarioDomain(**settings)


def test_membership_rejects_other_battery_horizons_ranges_and_publications():
    domain = ScenarioDomain(hours=2)
    scenario = sample_scenario(domain, Random(1))
    outsiders = [
        replace(scenario, config=replace(scenario.config, max_charge_kw=21)),
        sample_scenario(replace(domain, hours=3), Random(1)),
        replace(scenario, prices_eur_per_mwh=(201, 0)),
        replace(scenario, forecast_prices_eur_per_mwh=((201, 0), (0,))),
        replace(scenario, forecast_prices_eur_per_mwh=((), ())),
        replace(scenario, reserve_requirements_kwh=(91, 0)),
    ]
    for outsider in outsiders:
        with pytest.raises(ValueError):
            validate_scenario(outsider, domain)
        with pytest.raises(ValueError):
            mutate_scenario(outsider, domain, Random(1))


def test_fingerprint_ignores_seed_and_numeric_representation_only():
    domain = ScenarioDomain(price_min=0, price_max=0, reserve_max_kwh=0)
    scenario = sample_scenario(domain, Random(1))
    assert scenario_fingerprint(scenario) == scenario_fingerprint(replace(scenario, seed=2))
    integral = replace(scenario, prices_eur_per_mwh=(0,) * domain.hours)
    negative_zero = replace(scenario, prices_eur_per_mwh=(-0.0,) * domain.hours)
    assert scenario_fingerprint(scenario) == scenario_fingerprint(integral)
    assert scenario_fingerprint(scenario) == scenario_fingerprint(negative_zero)
    mutant = mutate_scenario(scenario, domain, Random(2))
    assert scenario_fingerprint(scenario) != scenario_fingerprint(mutant)


def test_mutating_actual_suffix_preserves_all_published_forecasts():
    domain = ScenarioDomain(hours=3)
    scenario = sample_scenario(domain, Random(1))
    altered = replace(scenario, prices_eur_per_mwh=(scenario.prices_eur_per_mwh[0], 0, 0))
    assert altered.forecast_prices_eur_per_mwh == scenario.forecast_prices_eur_per_mwh
    assert scenario_fingerprint(altered) != scenario_fingerprint(scenario)


def test_wide_finite_price_ranges_do_not_overflow_sampling():
    domain = ScenarioDomain(price_min=-1e308, price_max=1e308)
    validate_scenario(sample_scenario(domain, Random(1)), domain)
