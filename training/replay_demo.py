"""Demonstrate ViZDoom .lmp replay playback and index a representative success/failure sample."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import vizdoom as vd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env.spaces import FRAME_SKIP  # noqa: E402

IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
DEFAULT_EPISODES_CSV = Path(__file__).resolve().parent / "results" / "curriculum_eval_s42" / "episodes.csv"
DEFAULT_REPLAYS_DIR = Path(__file__).resolve().parent / "results" / "curriculum_eval_s42" / "replays"
DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent / "results" / "stage3_replay_sample"


def playback_lmp(wad_path: Path, map_name: str, lmp_path: Path, episode_ticks: int = 2100) -> dict:
    """Replay a recorded .lmp file in headless ViZDoom and return observed outcome."""
    g = vd.DoomGame()
    g.set_doom_game_path(IWAD)
    g.set_doom_scenario_path(str(wad_path.resolve()))
    g.set_doom_map(map_name)
    g.set_doom_skill(3)
    g.set_mode(vd.Mode.PLAYER)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_console_enabled(False)
    g.set_episode_timeout(episode_ticks)
    g.init()
    try:
        g.replay_episode(str(lmp_path.resolve()))
        steps = 0
        while not g.is_episode_finished():
            g.advance_action(FRAME_SKIP)
            steps += 1
        dead = bool(g.is_player_dead())
        timeout = bool(g.is_episode_timeout_reached())
        exit_reached = not dead and not timeout
        outcome = "exit" if exit_reached else ("death" if dead else "timeout")
        return {
            "lmp": lmp_path.name,
            "wad": str(wad_path.relative_to(ROOT)),
            "replay_steps": steps,
            "nominal_action_ticks": steps * FRAME_SKIP,
            "outcome": outcome,
            "exit_reached": exit_reached,
            "dead": dead,
            "timeout": timeout,
        }
    finally:
        g.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-csv", type=Path, default=DEFAULT_EPISODES_CSV)
    ap.add_argument("--replays-dir", type=Path, default=DEFAULT_REPLAYS_DIR)
    ap.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.episodes_csv.open()))
    # Pick 1 representative val/test success and 1 representative val/test failure
    success_row = next(r for r in rows if r["split"] in ("val", "test") and r["outcome"] == "exit")
    failure_row = next(r for r in rows if r["split"] in ("val", "test") and r["outcome"] != "exit")

    args.sample_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, row in (("representative_success", success_row), ("representative_failure", failure_row)):
        src_name = f"{row['layout']}_ep{row['episode_idx']}_seed{row['policy_seed']}.lmp"
        src_lmp = args.replays_dir / src_name
        assert src_lmp.is_file(), f"Missing source replay {src_lmp}"
        dst_lmp = args.sample_dir / src_name
        shutil.copy2(src_lmp, dst_lmp)
        playback = playback_lmp(ROOT / row["wad_path"], "MAP01", dst_lmp, episode_ticks=int(row["episode_ticks_cap"]))
        results.append(
            {
                "label": label,
                "layout": row["layout"],
                "split": row["split"],
                "episode_idx": int(row["episode_idx"]),
                "policy_seed": int(row["policy_seed"]),
                "recorded_decisions": int(row["decisions"]),
                "recorded_outcome": row["outcome"],
                "sample_file": str(dst_lmp.relative_to(ROOT)),
                "playback": playback,
            }
        )

    index_path = args.sample_dir / "index.json"
    index_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
