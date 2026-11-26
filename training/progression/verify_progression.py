"""Test-only 6-point scripted verification for the Stage 5A progression benchmark.

Verifies map SHA-256 hashes, D4 split hygiene, no bypass route, RAY_LOCKED
observation, locked door stays closed without key, key pickup, door opening
(ceiling - floor >= 56.0), and exit reachability. Measures whether the locked
door is sensed at key pickup (distinguishing occlusion from distance > 1024).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import platform
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import CONTRACT_VERSION, make_env  # noqa: E402
from doom_env.spaces import FRAME_SKIP, RAY_RANGE, VITALS_INDEX  # noqa: E402
from training.curriculum.verify_curriculum import check_split_hygiene  # noqa: E402
from training.generate_maps import CELL_SIZE  # noqa: E402
from training.progression.tracker import ProgressionEventTracker  # noqa: E402

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
DEFAULT_OUT_DIR = ROOT / "training" / "results" / "progression_verify_v2"

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
TURN_L = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
TURN_R = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8)
NOOP = np.zeros(9, dtype=np.int8)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def dependency_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("doom-env", "vizdoom", "gymnasium", "numpy", "torch"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def bfs_path(grid: list[str], start: tuple[int, int], goal: tuple[int, int], blocked: set[str]) -> list[tuple[int, int]] | None:
    h, w = len(grid), len(grid[0])
    q = deque([(start, [start])])
    seen = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] not in blocked and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), path + [(nr, nc)]))
    return None


def find_cell(grid: list[str], targets: set[str]) -> tuple[int, int]:
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch in targets:
                return (r, c)
    raise AssertionError(f"Missing target in {targets}")


def cell_center(r: int, c: int, h: int) -> tuple[float, float]:
    return ((c + 0.5) * CELL_SIZE, (h - 1 - r + 0.5) * CELL_SIZE)


def step_toward(
    env,
    obs: dict,
    tx: float,
    ty: float,
    stop_dist: float,
    max_steps: int,
    tracker: ProgressionEventTracker,
    door_sec: int,
    key_pickup_pos: list[tuple[float, float] | None],
):
    steps = 0
    while steps < max_steps:
        st = env.privileged_state()
        dx, dy = tx - st["x"], ty - st["y"]
        if math.hypot(dx, dy) <= stop_dist:
            return obs, False, False, {}, steps
        tracker.observe_pre_step(obs)
        target_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        diff = (target_deg - st["angle"] + 180.0) % 360.0 - 180.0
        act = TURN_L if diff > 12.0 else (TURN_R if diff < -12.0 else FWD)
        obs, _, term, trunc, info = env.step(act)
        st_post = env.privileged_state()
        was_picked = tracker.key_picked_up
        tracker.record_post_step(obs, st_post["sector_floor"], st_post["sector_ceiling"], bool(info.get("exit_reached", False)))
        if not was_picked and tracker.key_picked_up:
            key_pickup_pos[0] = (st_post["x"], st_post["y"])
        steps += 1
        if term or trunc:
            return obs, term, trunc, info, steps
    return obs, False, False, {}, steps


def follow_cells(
    env,
    obs: dict,
    grid: list[str],
    path: list[tuple[int, int]],
    stop_last_dist: float,
    tracker: ProgressionEventTracker,
    door_sec: int,
    key_pickup_pos: list[tuple[float, float] | None],
):
    h = len(grid)
    total = 0
    for idx in range(1, len(path)):
        tx, ty = cell_center(path[idx][0], path[idx][1], h)
        sdist = stop_last_dist if idx == len(path) - 1 else 20.0
        obs, term, trunc, info, d = step_toward(env, obs, tx, ty, sdist, 200, tracker, door_sec, key_pickup_pos)
        total += d
        if term or trunc:
            return obs, term, trunc, info, total
    return obs, False, False, {}, total


def verify_map(m: dict, replay_dir: Path | None = None) -> dict:
    wad_path = ROOT / m["wad_path"]
    actual_sha = sha256_file(wad_path)
    assert actual_sha == m["sha256"], f"Hash mismatch for {wad_path}: {actual_sha} != {m['sha256']}"

    grid = m["grid"]
    h = len(grid)
    p_cell = find_cell(grid, {"P"})
    k_cell = find_cell(grid, {"r", "b", "y"})
    d_cell = find_cell(grid, {"R", "B", "Y"})
    e_cell = find_cell(grid, {"E"})

    bypass = bfs_path(grid, p_cell, e_cell, blocked={"#", "R", "B", "Y"})
    assert bypass is None, f"{m['name']}: found bypass route {bypass}!"

    path_to_door = bfs_path(grid, p_cell, d_cell, blocked={"#", "E"})
    assert path_to_door is not None and len(path_to_door) >= 2
    approach_cell = path_to_door[-2]
    path_p_to_approach = path_to_door[:-1]
    path_approach_to_key = bfs_path(grid, approach_cell, k_cell, blocked={"#", "R", "B", "Y", "E"})
    assert path_approach_to_key is not None
    path_key_to_approach = list(reversed(path_approach_to_key))
    path_approach_to_exit = bfs_path(grid, approach_cell, e_cell, blocked={"#"})
    assert path_approach_to_exit is not None

    rec_path = (replay_dir / f"{m['name']}_verify.lmp") if replay_dir else None
    opts = {"record_path": str(rec_path)} if rec_path else None

    env = make_env(wad_path, m["map_name"], episode_ticks=2100, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0, options=opts)
        door_sec = int(np.where(env.privileged_state()["door_lock"] > 0)[0][0])
        key_vital_idx = VITALS_INDEX[f"key_{m['key_color']}_card"]
        tracker = ProgressionEventTracker(door_sec=door_sec, key_vital_idx=key_vital_idx)
        key_pickup_pos: list[tuple[float, float] | None] = [None]

        obs, term, trunc, info, _ = follow_cells(env, obs, grid, path_p_to_approach, 20.0, tracker, door_sec, key_pickup_pos)
        assert not (term or trunc)
        dx, dy = cell_center(d_cell[0], d_cell[1], h)
        obs, term, trunc, info, _ = step_toward(env, obs, dx, dy, 48.0, 40, tracker, door_sec, key_pickup_pos)
        assert tracker.door_observed_before_key, f"{m['name']}: RAY_LOCKED not observed before key"

        # Press USE without key
        tracker.observe_pre_step(obs)
        obs, _, term, trunc, info = env.step(USE)
        st = env.privileged_state()
        tracker.record_post_step(obs, st["sector_floor"], st["sector_ceiling"], bool(info.get("exit_reached", False)))
        for _ in range(10):
            tracker.observe_pre_step(obs)
            obs, _, term, trunc, info = env.step(NOOP)
            st = env.privileged_state()
            tracker.record_post_step(obs, st["sector_floor"], st["sector_ceiling"], bool(info.get("exit_reached", False)))
        assert float(st["sector_ceiling"][door_sec]) - float(st["sector_floor"][door_sec]) == 0.0, (
            f"{m['name']}: door opened without key!"
        )

        # Go collect matching key
        obs, term, trunc, info, _ = follow_cells(env, obs, grid, path_approach_to_key, 16.0, tracker, door_sec, key_pickup_pos)
        assert not (term or trunc) and tracker.key_picked_up

        # Return to door and unlock
        obs, term, trunc, info, _ = follow_cells(env, obs, grid, path_key_to_approach, 20.0, tracker, door_sec, key_pickup_pos)
        obs, term, trunc, info, _ = step_toward(env, obs, dx, dy, 48.0, 40, tracker, door_sec, key_pickup_pos)
        tracker.observe_pre_step(obs)
        obs, _, term, trunc, info = env.step(USE)
        st = env.privileged_state()
        tracker.record_post_step(obs, st["sector_floor"], st["sector_ceiling"], bool(info.get("exit_reached", False)))
        for _ in range(35):
            tracker.observe_pre_step(obs)
            obs, _, term, trunc, info = env.step(NOOP)
            st = env.privileged_state()
            tracker.record_post_step(obs, st["sector_floor"], st["sector_ceiling"], bool(info.get("exit_reached", False)))
            if tracker.door_passable_after_key:
                break
        assert tracker.door_passable_after_key, f"{m['name']}: door did not reach 56-unit clearance"

        # Walk through open door to exit
        obs, term, trunc, info, _ = follow_cells(env, obs, grid, path_approach_to_exit, 10.0, tracker, door_sec, key_pickup_pos)
        st = env.privileged_state()
        assert term and not trunc and tracker.ordered_sequence_complete and not st["dead"]

        assert key_pickup_pos[0] is not None
        door_dist = float(math.hypot(dx - key_pickup_pos[0][0], dy - key_pickup_pos[0][1]))
        if tracker.door_sensed_at_key_pickup:
            absence_reason = "sensed"
        elif door_dist > RAY_RANGE:
            absence_reason = "beyond_sensor_range"
        else:
            absence_reason = "occluded"

        return {
            "name": m["name"],
            "split": m["split"],
            "key_color": m["key_color"],
            "sha256": actual_sha,
            "no_bypass_verified": True,
            "stayed_locked_without_key": True,
            "door_first_observed_decision": tracker.door_first_observed_decision,
            "key_pickup_decision": tracker.key_pickup_decision,
            "door_sensed_at_key_pickup": tracker.door_sensed_at_key_pickup,
            "steps_door_absent_before_key": tracker.steps_door_absent_before_key,
            "door_distance_at_key_pickup": round(door_dist, 1),
            "ray_range": RAY_RANGE,
            "sensory_absence_reason": absence_reason,
            "opening_initiated_decision": tracker.opening_initiated_decision,
            "door_passable_decision": tracker.door_passable_decision,
            "ordered_sequence_complete": tracker.ordered_sequence_complete,
            "decisions": tracker.decision,
            "nominal_action_ticks": tracker.decision * FRAME_SKIP,
        }
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    maps = manifest["maps"]
    check_split_hygiene(maps)

    results = [verify_map(m, replay_dir=replay_dir if m["split"] == "val" else None) for m in maps]
    payload = {
        "contract_version": CONTRACT_VERSION,
        "command": " ".join(sys.argv),
        "git": git_revision(),
        "versions": dependency_versions(),
        "verified_count": len(results),
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
