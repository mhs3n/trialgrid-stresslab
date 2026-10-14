"""Compare fixed, random and targeted search with equal actual simulator-call caps."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from trialgrid import HoldController, Scenario
from trialgrid.controllers import BaselineConfig, ForecastOptimizer, HeuristicController
from trialgrid.controllers.llm import LLMController, canonical
from trialgrid.controllers.llm_models import FakeModel
from trialgrid.controllers.llm_types import LLMConfig
from trialgrid.metrics import FailureConfig
from trialgrid.search.domain import ScenarioDomain
from trialgrid.search.evaluation import (
    ControllerSpec,
    SearchConfig,
    run_search,
    verify_search_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=20_000)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--horizon", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--target", choices=("heuristic", "hold", "fake-llm"), default="heuristic")
    parser.add_argument("--economic-threshold-eur", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("search_results"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("search seeds must be distinct")
    directory = Path(__file__).parent / "scenarios" / "fixed"
    fixed = tuple(Scenario.from_json(path.read_text()) for path in sorted(directory.glob("*.json")))
    domain = ScenarioDomain(config=fixed[0].config, reserve_max_kwh=100)
    criteria = FailureConfig(economic_threshold_eur=args.economic_threshold_eur)
    config = BaselineConfig(horizon=args.horizon)
    if args.target == "heuristic":

        def factory():
            return HeuristicController(config)

        target_config = asdict(config)
    elif args.target == "fake-llm":
        llm_config = LLMConfig("fake-hold-v1", horizon=args.horizon, max_attempts=1)

        def factory():
            return LLMController(FakeModel(), llm_config)

        target_config = asdict(llm_config)
    else:
        factory = HoldController
        target_config = {"horizon": args.horizon}
    target = ControllerSpec(args.target, "1", args.horizon, canonical(target_config), factory)
    comparator = ControllerSpec(
        "forecast_optimizer",
        "1",
        args.horizon,
        canonical(asdict(config)),
        lambda: ForecastOptimizer(config),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "budget_definition": (
            "all M1 calls during evaluation, including planning and reserve witnesses"
        ),
        "verification_calls_in_search_budget": False,
        "criteria": asdict(criteria),
        "domain": domain.to_dict(),
        "target": json.loads(target.metadata_json()),
        "comparator": json.loads(comparator.metadata_json()),
        "runs": [],
        "paired_targeted_minus_random_unique_actionable": [],
        "claim": "descriptive synthetic experiment; no statistical superiority established",
    }
    for seed in args.seeds:
        summaries = {}
        for method in ("fixed", "random", "targeted"):
            budget = SearchConfig(args.calls, args.max_candidates, seed)
            started = perf_counter()
            result = run_search(
                method, domain, target, comparator, criteria, budget, fixed_scenarios=fixed
            )
            elapsed = perf_counter() - started
            payload = result.to_json()
            verify_search_result(payload)
            filename = f"{method}_seed_{seed}.json"
            (args.output_dir / filename).write_text(payload + "\n")
            summary = result.to_dict()["summary"]
            summaries[method] = summary
            manifest["runs"].append(
                {
                    "method": method,
                    "seed": seed,
                    "file": filename,
                    "elapsed_search_s": elapsed,
                    "summary": summary,
                }
            )
            print(
                f"seed={seed} {method}: calls={summary['used_simulator_calls']}/{args.calls}, "
                f"complete={summary['completed_evaluations']}, "
                f"unique actionable={summary['unique_actionable_cases']}"
            )
        difference = (
            summaries["targeted"]["unique_actionable_cases"]
            - summaries["random"]["unique_actionable_cases"]
        )
        manifest["paired_targeted_minus_random_unique_actionable"].append(
            {"seed": seed, "difference": difference}
        )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n"
    )
    print(f"Saved every candidate and verified derived evidence: {args.output_dir}")
    print(manifest["claim"])


if __name__ == "__main__":
    main()
