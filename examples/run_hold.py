"""Run the sample offline episode and export a verified JSON trace."""

import argparse
from pathlib import Path

from trialgrid import HoldController, Scenario, replay_episode, run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).parent / "scenarios" / "hold_24h.json",
    )
    parser.add_argument("--output", type=Path, default=Path("hold_episode.json"))
    args = parser.parse_args()
    scenario = Scenario.from_json(args.scenario.read_text())
    episode = run_episode(
        scenario, HoldController(), controller_name="hold", controller_version="1"
    )
    replay_episode(episode)
    args.output.write_text(episode.to_json() + "\n")
    print(
        f"Completed {len(episode.trace)} hours; profit EUR {episode.metrics.profit_eur:.2f}; "
        f"reserve misses {episode.metrics.reserve_violation_count}"
    )
    print(f"Verified episode trace: {args.output}")


if __name__ == "__main__":
    main()
