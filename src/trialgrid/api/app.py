"""Local-only FastAPI surface for creating and replaying verified evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator
from starlette.concurrency import run_in_threadpool

from trialgrid.api.controllers import build_controller
from trialgrid.metrics import FailureConfig
from trialgrid.results.artifacts import ArtifactStore, EvidenceArtifact, artifact_detail
from trialgrid.runner import run_episode
from trialgrid.scenario import Scenario, load_json

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_SCENARIO_HOURS = 168
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RequestSizeLimitMiddleware:
    """Reject oversized fixed or chunked bodies before request parsing."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        length = headers.get(b"content-length")
        if length is not None:
            try:
                if int(length) > self.max_bytes:
                    await self._reject(scope, receive, send, 413, "request exceeds size limit")
                    return
            except ValueError:
                await self._reject(scope, receive, send, 400, "invalid content-length")
                return
        messages = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                return
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    await self._reject(scope, receive, send, 413, "request exceeds size limit")
                    return
                if not message.get("more_body", False):
                    break

        async def replay_receive():
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(scope, receive, send, code: int, detail: str) -> None:
        response = JSONResponse(status_code=code, content={"detail": detail})
        await response(scope, receive, send)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scenario: dict
    target: Literal["hold", "heuristic", "forecast_optimizer"] = "heuristic"
    comparator: Literal["hold", "heuristic", "forecast_optimizer"] = "forecast_optimizer"
    horizon: StrictInt = 4
    economic_threshold_eur: float = 1.0

    @field_validator("economic_threshold_eur", mode="before")
    @classmethod
    def reject_boolean_threshold(cls, value):
        if isinstance(value, bool):
            raise ValueError("economic_threshold_eur must be numeric")
        return value


def _run(request: RunRequest, created_at_utc: str | None = None) -> EvidenceArtifact:
    scenario = Scenario.from_dict(request.scenario)
    if len(scenario.prices_eur_per_mwh) > MAX_SCENARIO_HOURS:
        raise ValueError(f"scenario cannot exceed {MAX_SCENARIO_HOURS} hours")
    criteria = FailureConfig(economic_threshold_eur=request.economic_threshold_eur)
    target, target_metadata = build_controller(request.target, request.horizon)
    comparator, comparator_metadata = build_controller(request.comparator, request.horizon)
    target_episode = run_episode(
        scenario, target, controller_name=request.target, controller_version="1"
    )
    comparator_episode = run_episode(
        scenario, comparator, controller_name=request.comparator, controller_version="1"
    )
    return EvidenceArtifact.create(
        scenario=scenario,
        target_metadata_json=target_metadata,
        comparator_metadata_json=comparator_metadata,
        criteria=criteria,
        target_episode=target_episode,
        comparator_episode=comparator_episode,
        created_at_utc=created_at_utc or datetime.now(UTC).isoformat(),
    )


def create_demo_artifact(store: ArtifactStore) -> EvidenceArtifact:
    scenario = Scenario.from_json(
        (PROJECT_ROOT / "examples/scenarios/fixed/price_cycle.json").read_text()
    )
    request = RunRequest(
        scenario=scenario.to_dict(),
        target="hold",
        comparator="forecast_optimizer",
        horizon=4,
        economic_threshold_eur=1.0,
    )
    artifact = _run(request, "2026-01-01T00:00:00+00:00")
    return store.save(artifact)


def create_app(
    artifact_root: Path | str,
    *,
    frontend_dist: Path | str | None = None,
) -> FastAPI:
    store = ArtifactStore(Path(artifact_root))
    app = FastAPI(
        title="TrialGrid evidence API",
        version="1",
        description="Local, unauthenticated, offline evidence replay for TrialGrid.",
    )
    app.state.artifact_store = store
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "network_model_calls": False}

    @app.get("/api/runs")
    async def list_runs() -> dict:
        return {"runs": await run_in_threadpool(store.list)}

    def load_or_error(artifact_id: str) -> EvidenceArtifact:
        try:
            return store.load(artifact_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="artifact not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @app.get("/api/runs/{artifact_id}")
    async def get_run(artifact_id: str) -> dict:
        artifact = await run_in_threadpool(load_or_error, artifact_id)
        return artifact_detail(artifact)

    @app.post("/api/runs/{artifact_id}/replay")
    async def replay_run(artifact_id: str) -> dict:
        artifact = await run_in_threadpool(load_or_error, artifact_id)
        return artifact_detail(artifact, replay_verified=True)

    @app.post("/api/runs/{artifact_id}/rerun", status_code=status.HTTP_201_CREATED)
    async def rerun_case(artifact_id: str) -> dict:
        artifact = await run_in_threadpool(load_or_error, artifact_id)
        target = load_json(artifact.target_metadata_json)
        comparator = load_json(artifact.comparator_metadata_json)
        try:
            request = RunRequest(
                scenario=artifact.scenario.to_dict(),
                target=target["name"],
                comparator=comparator["name"],
                horizon=target["horizon"],
                economic_threshold_eur=artifact.criteria.economic_threshold_eur,
            )
            rerun = await run_in_threadpool(_run, request)
            rerun = await run_in_threadpool(store.save, rerun)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return artifact_detail(rerun)

    @app.post("/api/runs", status_code=status.HTTP_201_CREATED)
    async def run_case(request: RunRequest) -> dict:
        try:
            artifact = await run_in_threadpool(_run, request)
            artifact = await run_in_threadpool(store.save, artifact)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return artifact_detail(artifact)

    @app.post("/api/demo", status_code=status.HTTP_201_CREATED)
    async def create_demo(response: Response) -> dict:
        try:
            artifact = await run_in_threadpool(create_demo_artifact, store)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        response.headers["Cache-Control"] = "no-store"
        return artifact_detail(artifact)

    static_root = (
        Path(frontend_dist) if frontend_dist is not None else PROJECT_ROOT / "frontend/dist"
    )
    if static_root.is_dir():
        app.mount("/", StaticFiles(directory=static_root, html=True), name="frontend")
    return app
