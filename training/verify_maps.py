"""Map reachability verification using scripted action sequences (test-only).

This script is ONLY a sanity check that both generated WADs load cleanly in
doom_env and that their walk-over exits (Exit_Normal, special 243) trigger.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from doom_env import make_env
from doom_env.spaces import FRAME_SKIP

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "maps" / "manifest.json"

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
TURN_L = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)


def git_revision() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def verify_practice(wad_path: Path, map_name: str) -> dict:
    env = make_env(wad_path, map_name, episode_ticks=2100)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs), "Observation outside official space"
        assert env.action_space.shape == (9,), f"Unexpected action shape {env.action_space.shape}"
        decisions = 0
        term = trunc = False
        while not (term or trunc) and decisions < 100:
            obs, _, term, trunc, info = env.step(FWD)
            decisions += 1
        st = env.privileged_state()
        assert term and not trunc and info["exit_reached"] and not st["dead"], (
            f"Practice exit failed: term={term}, trunc={trunc}, info={info}, dead={st['dead']}"
        )
        return {
            "name": "practice_corridor",
            "decisions": decisions,
            "nominal_action_ticks": decisions * FRAME_SKIP,
            "raw_engine_tic": int(info["tic"]),
            "exit_reached": bool(info["exit_reached"]),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "dead": bool(st["dead"]),
            "final_x": st["x"],
            "final_y": st["y"],
        }
    finally:
        env.close()


def verify_heldout(wad_path: Path, map_name: str) -> dict:
    env = make_env(wad_path, map_name, episode_ticks=2100)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs), "Observation outside official space"
        decisions = 0
        term = trunc = False
        while not (term or trunc) and decisions < 100:
            st = env.privileged_state()
            if st["x"] >= 270.0:
                break
            obs, _, term, trunc, info = env.step(FWD)
            decisions += 1
        while not (term or trunc) and decisions < 150:
            st = env.privileged_state()
            if 80.0 <= st["angle"] <= 100.0:
                break
            obs, _, term, trunc, info = env.step(TURN_L)
            decisions += 1
        while not (term or trunc) and decisions < 200:
            obs, _, term, trunc, info = env.step(FWD)
            decisions += 1
        st = env.privileged_state()
        assert term and not trunc and info["exit_reached"] and not st["dead"], (
            f"Held-out exit failed: term={term}, trunc={trunc}, info={info}, st={st}"
        )
        return {
            "name": "heldout_lshape",
            "decisions": decisions,
            "nominal_action_ticks": decisions * FRAME_SKIP,
            "raw_engine_tic": int(info["tic"]),
            "exit_reached": bool(info["exit_reached"]),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "dead": bool(st["dead"]),
            "final_x": st["x"],
            "final_y": st["y"],
        }
    finally:
        env.close()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    by_name = {m["name"]: m for m in manifest["maps"]}
    payload = {
        "git": git_revision(),
        "note": (
            "decisions is the primary duration metric; nominal_action_ticks = decisions * 4; "
            "raw_engine_tic is diagnostic (ViZDoom resets get_episode_time() to 0 on Exit_Normal)."
        ),
        "maps": [
            verify_practice(ROOT / by_name["practice_corridor"]["wad_path"], by_name["practice_corridor"]["map_name"]),
            verify_heldout(ROOT / by_name["heldout_lshape"]["wad_path"], by_name["heldout_lshape"]["map_name"]),
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
