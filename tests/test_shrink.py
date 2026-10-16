import pytest

from trialgrid.api.controllers import build_controller
from trialgrid.domain import BatteryConfig, BatteryState
from trialgrid.metrics import FailureConfig, FailureReport
from trialgrid.regression import (
    RegressionFixture,
    export_pytest_regression,
    run_regression_fixture,
)
from trialgrid.scenario import Scenario
from trialgrid.search.domain import ScenarioDomain
from trialgrid.search.evaluation import ControllerSpec, SearchConfig, run_search
from trialgrid.search.shrink import (
    ShrinkConfig,
    actionable_signature,
    shrink_failure,
    shrink_scenario,
)


def case() -> Scenario:
    return Scenario(
        1,
        77,
        BatteryConfig(100, 10, 90, 20, 20),
        BatteryState(50),
        (100, -100),
        ((100, -100), (-100,)),
        (60, 0),
    )


def specs() -> tuple[ControllerSpec, ControllerSpec]:
    return (
        ControllerSpec(
            "hold", "m2-hold-v1", 2, "{\"horizon\":2}", lambda: build_controller("hold", 2)[0]
        ),
        ControllerSpec(
            "forecast_optimizer",
            "m3-forecast-optimizer-v1",
            2,
            "{\"horizon\":2,\"power_fractions\":[0.5,1.0]}",
            lambda: build_controller("forecast_optimizer", 2)[0],
        ),
    )


def domain() -> ScenarioDomain:
    return ScenarioDomain(
        config=case().config,
        hours=2,
        forecast_horizon=2,
        reserve_min_kwh=0,
        reserve_max_kwh=90,
    )


def test_shrinker_preserves_actionable_signature_and_reduces_case():
    target, comparator = specs()
    criteria = FailureConfig()
    searched = run_search(
        "fixed",
        domain(),
        target,
        comparator,
        criteria,
        SearchConfig(simulator_call_budget=1000, max_candidates=1),
        fixed_scenarios=(case(),),
    )
    result = shrink_failure(
        searched.records[0], target, comparator, criteria, config=ShrinkConfig(200, 2)
    )
    assert result.complete
    assert result.changed
    assert result.signature == actionable_signature(result.minimized_report)
    assert result.original_scenario.seed == result.minimized_scenario.seed

    target_metadata = build_controller("hold", 2)[1]
    comparator_metadata = build_controller("forecast_optimizer", 2)[1]
    fixture = RegressionFixture.from_shrink(
        result, target_metadata, comparator_metadata, criteria
    )
    restored = RegressionFixture.from_json(fixture.to_json())
    assert restored == fixture
    assert actionable_signature(run_regression_fixture(restored)) == fixture.expected_signature


def test_exported_pytest_fixture_runs_offline(tmp_path, monkeypatch):
    target, comparator = specs()
    criteria = FailureConfig()
    searched = run_search(
        "fixed",
        domain(),
        target,
        comparator,
        criteria,
        SearchConfig(1000, 1),
        fixed_scenarios=(case(),),
    )
    result = shrink_failure(searched.records[0], target, comparator, criteria)
    fixture = RegressionFixture.from_shrink(
        result,
        build_controller("hold", 2)[1],
        build_controller("forecast_optimizer", 2)[1],
        criteria,
    )
    fixture_path, test_path = export_pytest_regression(
        fixture, tmp_path / "case.json", tmp_path / "test_case.py"
    )
    assert fixture_path.exists() and test_path.exists()
    monkeypatch.syspath_prepend(str(tmp_path))
    result = pytest.main([str(test_path), "-q"])
    assert result == pytest.ExitCode.OK


def test_non_actionable_and_incomplete_cases_are_rejected():
    # The record is actionable under the real case; an evaluator with no findings is not.
    with pytest.raises(ValueError, match="no actionable"):
        shrink_scenario(case(), lambda scenario: FailureReport((), 0, False, 50, 50))


def test_unknown_controller_metadata_is_rejected():
    target, comparator = specs()
    criteria = FailureConfig()
    searched = run_search(
        "fixed",
        domain(),
        target,
        comparator,
        criteria,
        SearchConfig(1000, 1),
        fixed_scenarios=(case(),),
    )
    result = shrink_failure(searched.records[0], target, comparator, criteria)
    fixture = RegressionFixture.from_shrink(
        result,
        build_controller("hold", 2)[1],
        build_controller("forecast_optimizer", 2)[1],
        criteria,
    )
    data = fixture.to_dict()
    data["target_metadata_json"] = data["target_metadata_json"].replace("m2-hold-v1", "unknown")
    with pytest.raises(ValueError, match="supported offline"):
        RegressionFixture.from_json(__import__("json").dumps(data))
