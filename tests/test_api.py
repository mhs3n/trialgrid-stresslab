"""The unauthenticated M7 API stays local, bounded and replay-first."""

from fastapi.testclient import TestClient

from trialgrid.api.app import create_app
from trialgrid.domain import BatteryConfig, BatteryState
from trialgrid.scenario import Scenario


def scenario() -> dict:
    return Scenario(
        1,
        77,
        BatteryConfig(100, 10, 90, 20, 20),
        BatteryState(50),
        (100, -100),
        ((100, -100), (-100,)),
        (60, 0),
    ).to_dict()


def client(tmp_path) -> TestClient:
    return TestClient(create_app(tmp_path / "artifacts", frontend_dist=tmp_path / "missing"))


def test_health_and_empty_run_list(tmp_path):
    api = client(tmp_path)
    assert api.get("/api/health").json() == {"status": "ok", "network_model_calls": False}
    assert api.get("/api/runs").json() == {"runs": []}


def test_run_list_detail_and_offline_replay(tmp_path, monkeypatch):
    def forbid_live_model(*args, **kwargs):
        raise AssertionError("live model construction is forbidden")

    monkeypatch.setattr(
        "trialgrid.controllers.llm_models.OpenAIModel", forbid_live_model
    )
    api = client(tmp_path)
    response = api.post(
        "/api/runs",
        json={
            "scenario": scenario(),
            "target": "hold",
            "comparator": "forecast_optimizer",
            "horizon": 2,
            "economic_threshold_eur": 1.0,
        },
    )
    assert response.status_code == 201
    detail = response.json()
    artifact_id = detail["id"]
    assert detail["replay_verified"] is True
    assert detail["report"]["findings"][0]["classification"] == "locally_preventable"

    listed = api.get("/api/runs").json()["runs"]
    assert listed[0]["id"] == artifact_id
    assert listed[0]["status"] == "verified"
    assert api.get(f"/api/runs/{artifact_id}").json() == detail
    replay = api.post(f"/api/runs/{artifact_id}/replay")
    assert replay.status_code == 200
    assert replay.json() == detail
    rerun = api.post(f"/api/runs/{artifact_id}/rerun")
    assert rerun.status_code == 201
    assert rerun.json()["id"] != artifact_id
    assert rerun.json()["target"]["metrics"] == detail["target"]["metrics"]
    assert rerun.json()["report"] == detail["report"]


def test_demo_is_deterministic_fixed_evidence(tmp_path):
    api = client(tmp_path)
    first = api.post("/api/demo")
    second = api.post("/api/demo")
    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()
    assert first.json()["scenario"]["hours"] == 24
    assert first.json()["target"]["metadata"]["name"] == "hold"
    assert first.json()["comparator"]["metadata"]["name"] == "forecast_optimizer"


def test_request_validation_and_bounds(tmp_path):
    api = client(tmp_path)
    payload = {
        "scenario": scenario(),
        "target": "unknown",
        "comparator": "forecast_optimizer",
        "horizon": 2,
        "economic_threshold_eur": 1.0,
    }
    assert api.post("/api/runs", json=payload).status_code == 422
    payload["target"] = "hold"
    payload["horizon"] = 7
    assert api.post("/api/runs", json=payload).status_code == 422
    payload["horizon"] = 2
    payload["economic_threshold_eur"] = True
    assert api.post("/api/runs", json=payload).status_code == 422
    assert api.get("/api/runs/not-an-id").status_code == 422


def test_oversized_request_is_rejected_before_parsing(tmp_path):
    api = client(tmp_path)
    response = api.post(
        "/api/runs",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": str(3 * 1024 * 1024)},
    )
    assert response.status_code == 413


def test_chunked_oversized_request_is_rejected_before_parsing(tmp_path):
    api = client(tmp_path)

    def chunks():
        yield b'{"scenario":"'
        yield b"x" * (2 * 1024 * 1024)
        yield b'"}'

    response = api.post(
        "/api/runs",
        content=chunks(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_tampered_saved_artifact_is_never_presented(tmp_path):
    api = client(tmp_path)
    created = api.post(
        "/api/runs",
        json={
            "scenario": scenario(),
            "target": "hold",
            "comparator": "forecast_optimizer",
            "horizon": 2,
            "economic_threshold_eur": 1.0,
        },
    ).json()
    path = tmp_path / "artifacts" / f"{created['id']}.json"
    text = path.read_text()
    path.write_text(text.replace('"profit_eur":0.0', '"profit_eur":1.0', 1))
    assert api.get(f"/api/runs/{created['id']}").status_code == 422
    assert api.get("/api/runs").json()["runs"] == [
        {"id": created["id"], "status": "invalid"}
    ]
