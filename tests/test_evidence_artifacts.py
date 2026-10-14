"""M7 evidence artifacts replay before presenting any derived values."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from trialgrid.api.app import RunRequest, _run
from trialgrid.domain import BatteryConfig, BatteryState
from trialgrid.results.artifacts import ArtifactStore, EvidenceArtifact, artifact_detail
from trialgrid.scenario import Scenario


def scenario() -> Scenario:
    return Scenario(
        1,
        77,
        BatteryConfig(100, 10, 90, 20, 20),
        BatteryState(50),
        (100, -100),
        ((100, -100), (-100,)),
        (60, 0),
    )


def artifact(created_at="2026-01-01T00:00:00+00:00") -> EvidenceArtifact:
    return _run(
        RunRequest(
            scenario=scenario().to_dict(),
            target="hold",
            comparator="forecast_optimizer",
            horizon=2,
            economic_threshold_eur=1.0,
        ),
        created_at,
    )


def test_round_trip_replays_episodes_report_and_ui_projection():
    value = artifact()
    restored = EvidenceArtifact.from_json(value.to_json())
    assert restored == value
    detail = artifact_detail(restored)
    assert detail["replay_verified"] is True
    assert detail["target"]["energy_points"] == [
        {"step": 0, "energy_kwh": 50},
        {"step": 1, "energy_kwh": 50},
        {"step": 2, "energy_kwh": 50},
    ]
    first = detail["target"]["steps"][0]
    assert first["requested_p_kw"] == first["applied_p_kw"] == 0
    assert first["reserve_shortfall_kwh"] == 10
    assert detail["report"]["findings"][0]["classification"] == "locally_preventable"
    assert "non-oracle" in detail["comparator"]["metadata"]["label"]


@pytest.mark.parametrize("corruption", ["episode", "report", "identity", "unknown"])
def test_tampered_artifact_is_rejected(corruption):
    data = artifact().to_dict()
    if corruption == "episode":
        data["target_episode"]["metrics"]["profit_eur"] += 1
    elif corruption == "report":
        data["report"]["profit_gap_eur"] += 1
    elif corruption == "identity":
        data["artifact_id"] = "0" * 24
    else:
        data["unexpected"] = True
    with pytest.raises(ValueError):
        EvidenceArtifact.from_json(json.dumps(data))


def test_duplicate_json_fields_are_rejected():
    text = artifact().to_json()
    duplicate = text[:-1] + ',"schema_version":1}'
    with pytest.raises(ValueError, match="duplicate"):
        EvidenceArtifact.from_json(duplicate)


def test_artifact_id_covers_timestamp_and_allows_repeated_runs():
    first = artifact("2026-01-01T00:00:00+00:00")
    second = artifact("2026-01-02T00:00:00+00:00")
    assert first.artifact_id != second.artifact_id


def test_store_load_list_limits_and_symlink_escape(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    value = store.save(artifact())
    assert store.load(value.artifact_id) == value
    assert store.list()[0]["status"] == "verified"

    tiny = ArtifactStore(tmp_path / "tiny", max_bytes=10)
    with pytest.raises(ValueError, match="size"):
        tiny.save(value)

    outside = tmp_path / "outside.json"
    outside.write_text(value.to_json())
    link = store.root / f"{'f' * 24}.json"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        store.load("f" * 24)
    assert any(item == {"id": "f" * 24, "status": "invalid"} for item in store.list())


def test_artifact_is_immutable():
    value = artifact()
    with pytest.raises(FrozenInstanceError):
        replace(value, artifact_id="bad").artifact_id = "worse"
