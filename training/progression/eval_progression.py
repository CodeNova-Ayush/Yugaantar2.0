"""Progression benchmark evaluator with explicit event-ordering metrics.

Uses ProgressionEventTracker (56-unit ceiling-minus-floor clearance) and reports
unconditional exit rates, unlock initiation rates, passable-door rates, and
ordered sequence rates with explicit numerators and denominators.
Defaults to train/val splits; pass 'test' explicitly only when frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import CONTRACT_VERSION, make_env  # noqa: E402
from doom_env.spaces import FRAME_SKIP, VITALS_INDEX  # noqa: E402
from submission.agent import DoomAgent  # noqa: E402
from training.eval_random import SeededRandomPolicy  # noqa: E402
from training.generate_maps import CELL_SIZE  # noqa: E402
from training.progression.tracker import DOOR_PROXIMITY_UNITS, ProgressionEventTracker, summarize_progression_episodes  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.json"
EPISODE_TICKS = 2100
ENV_SEEDS = list(range(20))
POLICY_SEEDS = [1000 + i for i in range(20)]


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


def _door_center_from_grid(grid: list[str]) -> tuple[float, float]:
    h = len(grid)
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch in ("R", "B", "Y"):
                return ((c + 0.5) * CELL_SIZE, (h - 1 - r + 0.5) * CELL_SIZE)
    raise ValueError("Grid has no locked door cell")


def evaluate_progression_layout(
    map_entry: dict,
    weights_path: Path | None = None,
    replay_dir: Path | None = None,
    clear_memory_each_step: bool = False,
) -> tuple[list[dict], dict]:
    wad_path = ROOT / map_entry["wad_path"]
    wad_sha256 = sha256_file(wad_path)
    assert wad_sha256 == map_entry["sha256"], f"Map SHA-256 mismatch for {wad_path}"

    env = make_env(wad_path, map_entry["map_name"], episode_ticks=EPISODE_TICKS)
    key_vital_idx = VITALS_INDEX[f"key_{map_entry['key_color']}_card"]
    door_center = _door_center_from_grid(map_entry["grid"])
    max_decisions = EPISODE_TICKS // FRAME_SKIP

    if weights_path is None:
        policy = SeededRandomPolicy(env.action_space.shape, seed=0)
    else:
        policy = DoomAgent(
            env.observation_space,
            env.action_space,
            "cpu",
            weights_path=weights_path,
            seed=0,
            clear_memory_each_step=clear_memory_each_step,
        )

    episodes: list[dict] = []
    try:
        for ep_idx, (env_seed, policy_seed) in enumerate(zip(ENV_SEEDS, POLICY_SEEDS)):
            if isinstance(policy, SeededRandomPolicy):
                policy.reset(seed=policy_seed)
            else:
                policy.seed = policy_seed
                policy.reset()

            rec_path = None
            if replay_dir is not None:
                replay_dir.mkdir(parents=True, exist_ok=True)
                rec_path = replay_dir / f"{map_entry['name']}_ep{ep_idx}_seed{policy_seed}.lmp"
            opts = {"record_path": str(rec_path)} if rec_path is not None else None

            obs, info = env.reset(seed=env_seed, options=opts)
            door_sec = int(np.where(env.privileged_state()["door_lock"] > 0)[0][0])
            tracker = ProgressionEventTracker(
                door_sec=door_sec,
                key_vital_idx=key_vital_idx,
                door_center=door_center,
                max_decisions=max_decisions,
            )

            total_reward = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                tracker.observe_pre_step(obs)
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                st = env.privileged_state()
                tracker.record_post_step(
                    obs,
                    st["sector_floor"],
                    st["sector_ceiling"],
                    bool(terminated and info.get("exit_reached", False)),
                    action=action,
                    player_pos=(st["x"], st["y"]),
                )

            outcome = "exit" if (terminated and info.get("exit_reached", False)) else ("death" if terminated else "timeout")
            episodes.append(
                {
                    "layout": map_entry["name"],
                    "split": map_entry["split"],
                    "key_color": map_entry["key_color"],
                    "wad_sha256": wad_sha256,
                    "episode_idx": ep_idx,
                    "env_seed": env_seed,
                    "policy_seed": policy_seed,
                    "decisions": tracker.decision,
                    "nominal_action_ticks": tracker.decision * FRAME_SKIP,
                    "outcome": outcome,
                    "failure_category": tracker.failure_category,
                    "exit_reached": bool(info.get("exit_reached", False)),
                    "door_first_observed_decision": tracker.door_first_observed_decision,
                    "door_observed_before_key": tracker.door_observed_before_key,
                    "key_picked_up": tracker.key_picked_up,
                    "key_pickup_decision": tracker.key_pickup_decision,
                    "remaining_budget_at_key": tracker.remaining_budget_at_key,
                    "door_sensed_at_key_pickup": tracker.door_sensed_at_key_pickup,
                    "steps_door_absent_before_key": tracker.steps_door_absent_before_key,
                    "door_resensed_after_key": tracker.door_resensed_after_key,
                    "door_resensed_decision": tracker.door_resensed_decision,
                    "returned_near_door_after_key": tracker.returned_near_door_after_key,
                    "returned_near_door_decision": tracker.returned_near_door_decision,
                    "min_door_distance_after_key": tracker.min_door_distance_after_key,
                    "use_rising_edges_near_door": tracker.use_rising_edges_near_door,
                    "opening_initiated_after_key": tracker.opening_initiated_after_key,
                    "opening_initiated_decision": tracker.opening_initiated_decision,
                    "door_reversals_after_key": tracker.door_reversals_after_key,
                    "door_passable_after_key": tracker.door_passable_after_key,
                    "door_passable_decision": tracker.door_passable_decision,
                    "exit_decision": tracker.exit_decision,
                    "ordered_sequence_complete": tracker.ordered_sequence_complete,
                    "env_reward": total_reward,
                }
            )
    finally:
        env.close()

    if replay_dir is not None:
        kept = []
        keep_paths = set()
        got_succ = got_fail = False
        for ep in episodes:
            p = replay_dir / f"{map_entry['name']}_ep{ep['episode_idx']}_seed{ep['policy_seed']}.lmp"
            if ep["outcome"] == "exit" and not got_succ and p.exists():
                got_succ = True
                keep_paths.add(p)
                kept.append({"outcome": ep["outcome"], "file": p.name, "decisions": ep["decisions"], "policy_seed": ep["policy_seed"]})
            elif ep["outcome"] != "exit" and not got_fail and p.exists():
                got_fail = True
                keep_paths.add(p)
                kept.append({"outcome": ep["outcome"], "file": p.name, "decisions": ep["decisions"], "policy_seed": ep["policy_seed"]})
        for ep in episodes:
            p = replay_dir / f"{map_entry['name']}_ep{ep['episode_idx']}_seed{ep['policy_seed']}.lmp"
            if p.exists() and p not in keep_paths:
                p.unlink()
        (replay_dir / f"{map_entry['name']}_replays.json").write_text(json.dumps(kept, indent=2) + "\n")

    summary = {
        "layout": map_entry["name"],
        "split": map_entry["split"],
        "key_color": map_entry["key_color"],
        "wad_sha256": wad_sha256,
        **summarize_progression_episodes(episodes),
    }
    return episodes, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--weights", type=Path, default=None, help="Omit for SeededRandomPolicy")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--save-replays-dir", type=Path, default=None)
    ap.add_argument("--clear-memory-each-step", action="store_true", help="Diagnostic ablation: zero external memory slots before each decision while keeping GRU state.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory {args.out_dir}")
    if (
        args.save_replays_dir is not None
        and args.save_replays_dir.exists()
        and any(args.save_replays_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(f"Refusing to overwrite non-empty replay directory {args.save_replays_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    selected = [m for m in manifest["maps"] if m["split"] in set(args.splits)]
    all_eps: list[dict] = []
    summaries: list[dict] = []
    for m in selected:
        eps, summ = evaluate_progression_layout(
            m,
            weights_path=args.weights,
            replay_dir=args.save_replays_dir,
            clear_memory_each_step=args.clear_memory_each_step,
        )
        all_eps.extend(eps)
        summaries.append(summ)

    with (args.out_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_eps[0].keys()))
        writer.writeheader()
        writer.writerows(all_eps)

    payload = {
        "contract_version": CONTRACT_VERSION,
        "command": " ".join(sys.argv),
        "git": git_revision(),
        "versions": dependency_versions(),
        "policy": str(args.weights) if args.weights else "SeededRandomPolicy",
        "checkpoint_sha256": sha256_file(args.weights) if args.weights else None,
        "clear_memory_each_step_ablation": bool(args.clear_memory_each_step),
        "splits": args.splits,
        "episode_ticks_cap": EPISODE_TICKS,
        "env_seeds": ENV_SEEDS,
        "policy_seeds": POLICY_SEEDS,
        "summaries": summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
