"""Compare observation-only baselines on one scenario and export replay evidence."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from trialgrid import Scenario, replay_episode, run_episode
from trialgrid.controllers import BaselineConfig, ForecastOptimizer, HeuristicController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).parent / "scenarios" / "hold_24h.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_results"))
    parser.add_argument("--horizon", type=int, default=4, choices=range(1, 7))
    args = parser.parse_args()
    scenario = Scenario.from_json(args.scenario.read_text())
    config = BaselineConfig(horizon=args.horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "information_mode": "published_forecast_only",
        "oracle": False,
        "controller_config": asdict(config),
        "reserve_policy": "minimize current shortfall; future reserves unknown",
        "optimizer_terminal_energy_value_eur_per_kwh": 0,
        "initial_energy_kwh": scenario.initial_state.energy_kwh,
        "runs": [],
    }
    print("Published-forecast-only comparison; no oracle or future realized prices.")
    print(f"Shared horizon cap: {config.horizon}; power fractions: {config.power_fractions}")
    print("Forecast optimizer: finite-action approximation, zero terminal energy value.")
    for name, controller in (
        ("heuristic", HeuristicController(config)),
        ("forecast_optimizer", ForecastOptimizer(config)),
    ):
        episode = run_episode(scenario, controller, controller_name=name, controller_version="1")
        replay_episode(episode)
        filename = f"{name}.json"
        (args.output_dir / filename).write_text(episode.to_json() + "\n")
        final_energy = episode.trace[-1].result.state.energy_kwh
        report["runs"].append(
            {
                "controller_name": name,
                "controller_version": "1",
                "episode_file": filename,
                "metrics": asdict(episode.metrics),
                "final_energy_kwh": final_energy,
            }
        )
        print(
            f"{name}: realized profit EUR {episode.metrics.profit_eur:.4f}, "
            f"final energy {final_energy:.4f} kWh, "
            f"reserve misses {episode.metrics.reserve_violation_count}, "
            f"rejected actions {episode.metrics.rejected_action_count}"
        )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    )
    print(f"Verified traces and shared configuration: {args.output_dir}")
    print("Illustrative scenario only; these results do not establish controller superiority.")


if __name__ == "__main__":
    main()
