"""Offline fake by default; optional OpenAI dispatch or recorded-response replay."""

import argparse
import json
import os
from pathlib import Path

from trialgrid import Scenario
from trialgrid.controllers.llm_audit import LLMRun, run_llm_episode, summarize_llm_run
from trialgrid.controllers.llm_models import FakeModel, OpenAIModel
from trialgrid.controllers.llm_types import LLMConfig, TokenPricing


def environment_pricing() -> TokenPricing | None:
    names = (
        "TRIALGRID_INPUT_USD_PER_MILLION",
        "TRIALGRID_CACHED_INPUT_USD_PER_MILLION",
        "TRIALGRID_OUTPUT_USD_PER_MILLION",
        "TRIALGRID_PRICING_SOURCE",
    )
    values = [os.environ.get(name) for name in names]
    if not any(value is not None for value in values):
        return None
    if not all(value is not None and value.strip() for value in values):
        raise ValueError("supply all three token tariffs and TRIALGRID_PRICING_SOURCE, or none")
    return TokenPricing(*(float(value) for value in values[:3]), values[3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="use environment-configured OpenAI model")
    mode.add_argument("--replay", type=Path, help="verify saved responses/errors with no API calls")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).parent / "scenarios" / "hold_24h.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("llm_results"))
    parser.add_argument("--horizon", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--max-attempts", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--failure-policy", choices=("raise", "hold"), default="raise")
    args = parser.parse_args()
    if args.replay:
        run = LLMRun.from_json(args.replay.read_text())
        execution_mode = "recorded_replay"
    else:
        scenario = Scenario.from_json(args.scenario.read_text())
        if args.real:
            model_id = os.environ.get("TRIALGRID_LLM_MODEL", "").strip()
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not model_id or not api_key:
                parser.error("--real requires TRIALGRID_LLM_MODEL and OPENAI_API_KEY")
            pricing = environment_pricing()
            model = OpenAIModel(
                api_key,
                timeout_s=float(os.environ.get("TRIALGRID_LLM_TIMEOUT_S", "30")),
            )
            execution_mode = "openai"
        else:
            model_id = "fake-hold-v1"
            model = FakeModel()
            pricing = TokenPricing(0, 0, 0, "offline synthetic fake: no API calls")
            execution_mode = "fake"
        config = LLMConfig(
            model_id=model_id,
            horizon=args.horizon,
            max_attempts=args.max_attempts,
            failure_policy=args.failure_policy,
            retry_delay_s=0.5 if args.real else 0,
            pricing=pricing,
        )
        run = run_llm_episode(scenario, model, config)
        # Verify the complete audit, not just recorded dispatch actions.
        LLMRun.from_json(run.to_json())
    report = summarize_llm_run(run)
    report["execution_mode"] = execution_mode
    report["new_api_calls"] = report["call_attempts"] if args.real else 0
    if not args.real:
        report["new_api_cost_usd"] = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "llm_run.json").write_text(run.to_json() + "\n")
    (args.output_dir / "episode.json").write_text(
        (run.episode.to_json() if run.episode is not None else "null") + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True, indent=2))
    print(f"Verified response/error replay; audit saved in {args.output_dir}")
    return 0 if run.episode is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
