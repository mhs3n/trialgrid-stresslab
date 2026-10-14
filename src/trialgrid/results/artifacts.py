"""Versioned, replay-verified evidence stored as local content-addressed JSON."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

from trialgrid.controllers.llm import canonical
from trialgrid.metrics import FailureConfig, FailureFinding, FailureReport, evaluate_episode
from trialgrid.runner import Episode, _matches_json
from trialgrid.scenario import Scenario, load_json, require_keys

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_ID = re.compile(r"[0-9a-f]{24}")
DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024


def _metadata(text: str) -> dict:
    value = require_keys(
        load_json(text),
        {"name", "version", "horizon", "configuration", "information_mode"},
    )
    if any(
        not isinstance(value[key], str) or not value[key].strip()
        for key in ("name", "version")
    ):
        raise ValueError("controller name/version must be nonempty")
    if type(value["horizon"]) is not int or not 1 <= value["horizon"] <= 6:
        raise ValueError("controller horizon must be in [1, 6]")
    if not isinstance(value["configuration"], dict):
        raise ValueError("controller configuration must be an object")
    if value["information_mode"] != "published_forecast_only":
        raise ValueError("unsupported controller information mode")
    return value


def _timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("created_at_utc must be text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("created_at_utc must be ISO 8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("created_at_utc must include a UTC offset")
    return value


def _report_from_dict(value: dict) -> FailureReport:
    data = require_keys(value, {field.name for field in fields(FailureReport)})
    if not isinstance(data["findings"], list):
        raise ValueError("findings must be an array")
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


def _artifact_id(payload: dict) -> str:
    identity = dict(payload)
    identity.pop("artifact_id", None)
    return hashlib.sha256(canonical(identity).encode()).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    artifact_id: str
    created_at_utc: str
    scenario: Scenario
    target_metadata_json: str
    comparator_metadata_json: str
    criteria: FailureConfig
    target_episode: Episode
    comparator_episode: Episode
    report: FailureReport
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return canonical(asdict(self))

    @classmethod
    def create(
        cls,
        *,
        scenario: Scenario,
        target_metadata_json: str,
        comparator_metadata_json: str,
        criteria: FailureConfig,
        target_episode: Episode,
        comparator_episode: Episode,
        created_at_utc: str | None = None,
    ) -> EvidenceArtifact:
        report = evaluate_episode(target_episode, comparator_episode, criteria)
        provisional = {
            "artifact_id": "",
            "created_at_utc": created_at_utc or datetime.now(UTC).isoformat(),
            "scenario": scenario.to_dict(),
            "target_metadata_json": target_metadata_json,
            "comparator_metadata_json": comparator_metadata_json,
            "criteria": asdict(criteria),
            "target_episode": target_episode.to_dict(),
            "comparator_episode": comparator_episode.to_dict(),
            "report": asdict(report),
            "schema_version": ARTIFACT_SCHEMA_VERSION,
        }
        provisional["artifact_id"] = _artifact_id(provisional)
        return cls.from_dict(json.loads(canonical(provisional)))

    @classmethod
    def from_json(cls, text: str) -> EvidenceArtifact:
        return cls.from_dict(load_json(text))

    @classmethod
    def from_dict(cls, value: dict) -> EvidenceArtifact:
        data = require_keys(value, {field.name for field in fields(cls)})
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported evidence schema_version")
        if not isinstance(data["artifact_id"], str) or not ARTIFACT_ID.fullmatch(
            data["artifact_id"]
        ):
            raise ValueError("invalid artifact_id")
        _timestamp(data["created_at_utc"])
        scenario = Scenario.from_dict(data["scenario"])
        target_metadata = _metadata(data["target_metadata_json"])
        comparator_metadata = _metadata(data["comparator_metadata_json"])
        if target_metadata["horizon"] != comparator_metadata["horizon"]:
            raise ValueError("controller information horizons differ")
        criteria = FailureConfig(
            **require_keys(data["criteria"], {field.name for field in fields(FailureConfig)})
        )
        target_episode = Episode.from_dict(data["target_episode"])
        comparator_episode = Episode.from_dict(data["comparator_episode"])
        for episode, metadata in (
            (target_episode, target_metadata),
            (comparator_episode, comparator_metadata),
        ):
            if episode.scenario != scenario:
                raise ValueError("episode scenario does not match artifact")
            if (
                episode.controller_name != metadata["name"]
                or episode.controller_version != metadata["version"]
            ):
                raise ValueError("episode controller identity does not match metadata")
        report = evaluate_episode(target_episode, comparator_episode, criteria)
        supplied_report = _report_from_dict(data["report"])
        if not _matches_json(asdict(supplied_report), asdict(report)):
            raise ValueError("failure report does not match episode evidence")
        if data["artifact_id"] != _artifact_id(data):
            raise ValueError("artifact_id does not match evidence")
        return cls(
            data["artifact_id"],
            data["created_at_utc"],
            scenario,
            data["target_metadata_json"],
            data["comparator_metadata_json"],
            criteria,
            target_episode,
            comparator_episode,
            report,
        )


def _episode_view(episode: Episode, metadata_json: str) -> dict:
    metadata = _metadata(metadata_json)
    energy_points = [{"step": 0, "energy_kwh": episode.scenario.initial_state.energy_kwh}]
    steps = []
    for trace in episode.trace:
        current = trace.observation.current
        energy_points.append(
            {"step": current.step_index + 1, "energy_kwh": trace.result.state.energy_kwh}
        )
        steps.append(
            {
                "step": current.step_index,
                "actual_price_eur_per_mwh": current.price_eur_per_mwh,
                "published_forecast_eur_per_mwh": list(
                    current.forecast_prices_eur_per_mwh
                ),
                "reserve_requirement_kwh": current.reserve_requirement_kwh,
                "start_energy_kwh": current.state.energy_kwh,
                "end_energy_kwh": trace.result.state.energy_kwh,
                "requested_p_kw": trace.result.requested_action.p_kw,
                "applied_p_kw": trace.result.applied_action.p_kw,
                "revenue_eur": trace.result.revenue_eur,
                "degradation_cost_eur": trace.result.degradation_cost_eur,
                "profit_eur": trace.result.profit_eur,
                "physical_violations": list(trace.result.violations),
                "reserve_shortfall_kwh": trace.reserve_shortfall_kwh,
                "violations": list(trace.violations),
            }
        )
    label = metadata["name"]
    if metadata["name"] == "forecast_optimizer":
        label = "Forecast-only finite-action optimizer (non-oracle)"
    metrics = asdict(episode.metrics)
    metrics["ending_energy_kwh"] = episode.trace[-1].result.state.energy_kwh
    return {
        "metadata": {**metadata, "label": label},
        "metrics": metrics,
        "energy_points": energy_points,
        "steps": steps,
    }


def artifact_detail(artifact: EvidenceArtifact, *, replay_verified: bool = True) -> dict:
    return {
        "id": artifact.artifact_id,
        "created_at_utc": artifact.created_at_utc,
        "replay_verified": replay_verified,
        "scenario": {
            "hours": len(artifact.scenario.prices_eur_per_mwh),
            "seed": artifact.scenario.seed,
            "initial_energy_kwh": artifact.scenario.initial_state.energy_kwh,
            "energy_floor_kwh": artifact.scenario.config.energy_floor_kwh,
            "energy_ceiling_kwh": artifact.scenario.config.energy_ceiling_kwh,
        },
        "target": _episode_view(artifact.target_episode, artifact.target_metadata_json),
        "comparator": _episode_view(
            artifact.comparator_episode, artifact.comparator_metadata_json
        ),
        "report": asdict(artifact.report),
        "notices": [
            "Replay verifies internal consistency, not artifact authenticity.",
            "Positive power discharges; negative power charges.",
            "Reserve requirements are post-step energy commitments.",
            "M6 shrinking and pytest regression export are not included.",
        ],
    }


def artifact_summary(artifact: EvidenceArtifact) -> dict:
    target = artifact.target_episode
    comparator = artifact.comparator_episode
    return {
        "id": artifact.artifact_id,
        "created_at_utc": artifact.created_at_utc,
        "target_name": _metadata(artifact.target_metadata_json)["name"],
        "comparator_name": _metadata(artifact.comparator_metadata_json)["name"],
        "hours": len(artifact.scenario.prices_eur_per_mwh),
        "target_profit_eur": target.metrics.profit_eur,
        "comparator_profit_eur": comparator.metrics.profit_eur,
        "target_reserve_violations": target.metrics.reserve_violation_count,
        "target_rejected_actions": target.metrics.rejected_action_count,
        "finding_count": len(artifact.report.findings),
        "status": "verified",
    }


class ArtifactStore:
    """A single configured directory addressed only by validated content IDs."""

    def __init__(self, root: Path, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _path(self, artifact_id: str) -> Path:
        if not isinstance(artifact_id, str) or not ARTIFACT_ID.fullmatch(artifact_id):
            raise ValueError("invalid artifact id")
        path = self.root / f"{artifact_id}.json"
        if path.is_symlink():
            raise ValueError("artifact symlinks are not allowed")
        if path.resolve(strict=False).parent != self.root:
            raise ValueError("artifact escapes configured root")
        return path

    def save(self, artifact: EvidenceArtifact) -> EvidenceArtifact:
        text = artifact.to_json() + "\n"
        if len(text.encode()) > self.max_bytes:
            raise ValueError("artifact exceeds size limit")
        path = self._path(artifact.artifact_id)
        if path.exists():
            existing = self.load(artifact.artifact_id)
            if existing.to_json() != artifact.to_json():
                raise ValueError("artifact id collision")
            return existing
        handle, temporary = tempfile.mkstemp(prefix=".trialgrid-", dir=self.root)
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return artifact

    def load(self, artifact_id: str) -> EvidenceArtifact:
        path = self._path(artifact_id)
        if not path.is_file():
            raise FileNotFoundError(artifact_id)
        if path.stat().st_size > self.max_bytes:
            raise ValueError("artifact exceeds size limit")
        return EvidenceArtifact.from_json(path.read_text())

    def list(self) -> list[dict]:
        summaries = []
        for path in sorted(self.root.glob("*.json")):
            if not ARTIFACT_ID.fullmatch(path.stem):
                continue
            try:
                summaries.append(artifact_summary(self.load(path.stem)))
            except (OSError, ValueError):
                summaries.append({"id": path.stem, "status": "invalid"})
        return sorted(summaries, key=lambda item: item.get("created_at_utc", ""), reverse=True)
