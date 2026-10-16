"""Offline JSON and pytest regression fixtures for minimized failures."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from trialgrid.api.controllers import CONTROLLER_VERSIONS, build_controller
from trialgrid.metrics import FailureConfig, FailureFinding, FailureReport, evaluate_episode
from trialgrid.runner import Episode, run_episode
from trialgrid.scenario import Scenario, load_json, require_keys
from trialgrid.search.shrink import FailureSignature, ShrinkResult, actionable_signature

REGRESSION_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _report(value: dict) -> FailureReport:
    data = require_keys(value, {field.name for field in fields(FailureReport)})
    findings = tuple(
        FailureFinding(**require_keys(item, {field.name for field in fields(FailureFinding)}))
        for item in data["findings"]
    )
    return FailureReport(
        findings,
        data["profit_gap_eur"],
        data["economic_eligible"],
        data["tested_ending_energy_kwh"],
        data["comparator_ending_energy_kwh"],
    )


@dataclass(frozen=True, slots=True)
class RegressionFixture:
    scenario: Scenario
    target_metadata_json: str
    comparator_metadata_json: str
    criteria: FailureConfig
    expected_signature: FailureSignature
    repetitions: int = 1
    schema_version: int = REGRESSION_SCHEMA_VERSION

    @classmethod
    def from_shrink(
        cls,
        result: ShrinkResult,
        target_metadata_json: str,
        comparator_metadata_json: str,
        criteria: FailureConfig,
    ) -> RegressionFixture:
        if not result.complete:
            raise ValueError("cannot export an incomplete shrink result")
        _controller_metadata(target_metadata_json)
        _controller_metadata(comparator_metadata_json)
        if not result.signature:
            raise ValueError("cannot export an empty failure signature")
        return cls(
            result.minimized_scenario,
            target_metadata_json,
            comparator_metadata_json,
            criteria,
            result.signature,
            result.repetitions,
        )

    def to_dict(self) -> dict:
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return json.dumps(
            asdict(self), default=_json_default, allow_nan=False, sort_keys=True, indent=2
        )

    @classmethod
    def from_json(cls, text: str) -> RegressionFixture:
        data = require_keys(load_json(text), {field.name for field in fields(cls)})
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported regression schema_version")
        if not isinstance(data["expected_signature"], list):
            raise ValueError("invalid expected failure signature")
        scenario = Scenario.from_dict(data["scenario"])
        criteria = FailureConfig(
            **require_keys(data["criteria"], {field.name for field in fields(FailureConfig)})
        )
        signature = tuple(
            tuple(item)
            for item in data["expected_signature"]
            if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item)
        )
        if not signature or len(signature) != len(data["expected_signature"]):
            raise ValueError("invalid expected failure signature")
        if type(data["repetitions"]) is not int or not 1 <= data["repetitions"] <= 20:
            raise ValueError("invalid regression repetitions")
        _controller_metadata(data["target_metadata_json"])
        _controller_metadata(data["comparator_metadata_json"])
        return cls(
            scenario,
            data["target_metadata_json"],
            data["comparator_metadata_json"],
            criteria,
            signature,
            data["repetitions"],
        )


def _json_default(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _controller_metadata(text: str) -> dict:
    data = require_keys(
        load_json(text), {"name", "version", "horizon", "configuration", "information_mode"}
    )
    if not isinstance(data["name"], str) or not isinstance(data["version"], str):
        raise ValueError("controller name/version must be text")
    if (
        data["name"] not in CONTROLLER_VERSIONS
        or data["version"] != CONTROLLER_VERSIONS[data["name"]]
    ):
        raise ValueError("regression fixture requires a supported offline controller version")
    if not isinstance(data["configuration"], dict):
        raise ValueError("controller configuration must be an object")
    if type(data["horizon"]) is not int or not 1 <= data["horizon"] <= 6:
        raise ValueError("invalid controller horizon")
    if data["information_mode"] != "published_forecast_only":
        raise ValueError("unsupported controller information mode")
    return data


def _run_controller(metadata_json: str, scenario: Scenario) -> Episode:
    metadata = _controller_metadata(metadata_json)
    controller, actual_metadata = build_controller(metadata["name"], metadata["horizon"])
    if load_json(actual_metadata) != metadata:
        raise ValueError("controller metadata does not match the allowlisted implementation")
    return run_episode(
        scenario,
        controller,
        controller_name=metadata["name"],
        controller_version=metadata["version"],
    )


def run_regression_fixture(fixture: RegressionFixture) -> FailureReport:
    """Run an offline fixture and return its verified report."""
    reports = []
    for _ in range(fixture.repetitions):
        target = _run_controller(fixture.target_metadata_json, fixture.scenario)
        comparator = _run_controller(fixture.comparator_metadata_json, fixture.scenario)
        reports.append(evaluate_episode(target, comparator, fixture.criteria))
    expected = fixture.expected_signature
    if any(actionable_signature(report) != expected for report in reports):
        raise AssertionError("regression failure predicate no longer reproduces")
    return reports[0]


def assert_regression_fixture(fixture: RegressionFixture) -> None:
    run_regression_fixture(fixture)


def export_pytest_regression(
    fixture: RegressionFixture, fixture_path: Path | str, test_path: Path | str
) -> tuple[Path, Path]:
    """Write a JSON fixture and a portable offline pytest module."""
    fixture_path = Path(fixture_path)
    test_path = Path(test_path)
    if fixture_path.suffix != ".json" or test_path.suffix != ".py":
        raise ValueError("regression paths must end in .json and .py")
    if fixture_path.parent != test_path.parent:
        raise ValueError("fixture and generated test must share a directory")
    for path in (fixture_path, test_path):
        if not _SAFE_NAME.fullmatch(path.name):
            raise ValueError("unsafe regression filename")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(fixture.to_json() + "\n")
    test_path.write_text(
        "from pathlib import Path\n"
        "from trialgrid.regression import RegressionFixture, assert_regression_fixture\n\n"
        f"def test_regression_case():\n"
        f"    fixture = RegressionFixture.from_json(\n"
        f"        Path(__file__).with_name({fixture_path.name!r}).read_text()\n"
        f"    )\n"
        "    assert_regression_fixture(fixture)\n"
    )
    return fixture_path, test_path

