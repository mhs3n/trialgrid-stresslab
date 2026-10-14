"""Search preserves evidence and charges all simulation work to finite budgets."""

import json
from dataclasses import replace
from random import Random

import pytest

from trialgrid.controllers import BaselineConfig, ForecastOptimizer
from trialgrid.controllers.llm import ControllerCallFailure
from trialgrid.metrics import FailureConfig
from trialgrid.runner import HoldController
from trialgrid.search.domain import ScenarioDomain, sample_scenario
from trialgrid.search.evaluation import (
    ControllerSpec,
    SearchConfig,
    run_search,
    verify_search_result,
)


def spec(factory=HoldController, *, horizon=2):
    return ControllerSpec('test', '1', horizon, '{}', factory)


def domain(**kwargs):
    return ScenarioDomain(hours=2, forecast_horizon=2, reserve_max_kwh=0, **kwargs)


def run(*, method='random', space=None, target=None, comparator=None, budget=None, fixed=()):
    return run_search(method, space or domain(), target or spec(), comparator or spec(),
                      FailureConfig(), budget or SearchConfig(1000, 3, 1), fixed_scenarios=fixed)


def test_random_deterministic_fresh_controllers_and_negative_evidence():
    created = []

    class FreshHold:
        def __init__(self):
            self.calls = 0
            created.append(self)

        def act(self, observation, rng):
            assert self.calls == observation.current.step_index
            self.calls += 1
            return HoldController().act(observation, rng)

    result = run(target=spec(FreshHold), comparator=spec(FreshHold))
    assert len(created) == 6
    assert all(controller.calls == 2 for controller in created)
    assert len(result.records) == 3
    assert all(
        record.report is not None and not record.report.findings for record in result.records
    )
    assert result.to_json() == run().to_json()
    assert verify_search_result(result.to_json()) == result.to_dict()
    assert result.to_dict()['summary']['stop_reason'] == 'candidate_cap'


def test_fixed_duplicates_consume_budget_and_deduplicate_summary():
    space = domain()
    scenario = sample_scenario(space, Random(1))
    result = run(method='fixed', space=space, fixed=(scenario,), budget=SearchConfig(12, 10, 1))
    assert [record.simulator_calls for record in result.records] == [4, 4, 4]
    summary = result.to_dict()['summary']
    assert summary['attempted_candidates'] == 3
    assert summary['unique_environments'] == 1
    assert summary['used_simulator_calls'] == 12
    assert summary['stop_reason'] == 'simulator_budget'


@pytest.mark.parametrize('limit,expected', [(1, (1,)), (3, (3,)), (5, (4, 1))])
def test_exact_budget_keeps_interrupted_candidate(limit, expected):
    result = run(budget=SearchConfig(limit, 10, 1))
    assert tuple(record.simulator_calls for record in result.records) == expected
    assert result.records[-1].cumulative_calls == limit
    assert result.records[-1].report is None
    assert 'simulator_budget_exhausted' in (
        result.records[-1].target_error, result.records[-1].comparator_error)
    assert verify_search_result(result.to_json()) == result.to_dict()


def test_optimizer_internal_planning_is_counted_and_bounded():
    optimizer = spec(lambda: ForecastOptimizer(BaselineConfig(horizon=2)))
    complete = run(target=optimizer, budget=SearchConfig(10000, 1, 1))
    assert complete.records[0].simulator_calls > 4
    limited = run(target=optimizer, budget=SearchConfig(3, 2, 1))
    assert limited.records[0].simulator_calls == 3
    assert limited.records[0].target_error == 'simulator_budget_exhausted'
    assert limited.records[0].target_episode is None


def test_reserve_classification_witnesses_are_counted_and_can_interrupt():
    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_min_kwh=90, reserve_max_kwh=90)
    complete = run(space=space, budget=SearchConfig(10000, 1, 1))
    assert complete.records[0].simulator_calls > 4
    limited = run(space=space, budget=SearchConfig(5, 2, 1))
    record = limited.records[0]
    assert record.target_episode is not None and record.comparator_episode is not None
    assert record.evaluation_error == 'simulator_budget_exhausted'
    assert record.report is None and record.simulator_calls == 5
    assert verify_search_result(limited.to_json()) == limited.to_dict()


def test_error_categories_retained_provider_messages_excluded_and_zero_work_capped():
    def model_failure():
        raise ControllerCallFailure('private provider text')

    def generic_failure():
        raise RuntimeError('secret-key-do-not-save')

    result = run(target=spec(model_failure), comparator=spec(generic_failure),
                 budget=SearchConfig(10, 3, 1))
    assert len(result.records) == 3
    assert all(record.target_error == 'model_call_failure' for record in result.records)
    assert all(record.comparator_error == 'controller_error:RuntimeError'
               for record in result.records)
    assert all(record.simulator_calls == 0 and record.report is None for record in result.records)
    assert 'private provider' not in result.to_json()
    assert 'secret-key' not in result.to_json()
    assert result.to_dict()['summary']['stop_reason'] == 'candidate_cap'
    assert verify_search_result(result.to_json()) == result.to_dict()


def test_horizon_mismatch_and_missing_fixed_suite_rejected():
    with pytest.raises(ValueError, match='horizon'):
        run(target=spec(horizon=1))
    with pytest.raises(ValueError, match='fixed'):
        run(method='fixed')
    space = domain()
    outside = sample_scenario(replace(space, hours=3), Random(1))
    with pytest.raises(ValueError):
        run(method='fixed', fixed=(outside,))


@pytest.mark.parametrize('corruption', ['summary', 'episode', 'fingerprint', 'report', 'calls'])
def test_artifact_corruption_is_rejected(corruption):
    data = run().to_dict()
    record = data['records'][0]
    if corruption == 'summary':
        data['summary']['completed_evaluations'] += 1
    elif corruption == 'episode':
        record['target_episode']['metrics']['profit_eur'] += 1
    elif corruption == 'fingerprint':
        record['scenario_id'] = 'bad'
    elif corruption == 'report':
        record['report']['profit_gap_eur'] += 1
    else:
        record['cumulative_calls'] += 1
    with pytest.raises(ValueError):
        verify_search_result(json.dumps(data))


@pytest.mark.parametrize('settings', [
    {'simulator_call_budget': 0}, {'simulator_call_budget': True},
    {'max_candidates': 0}, {'max_candidates': 1.0}, {'seed': -1}, {'seed': 2**64},
])
def test_invalid_budgets(settings):
    with pytest.raises(ValueError):
        SearchConfig(**settings)
