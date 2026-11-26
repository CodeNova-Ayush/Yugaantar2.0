"""Evaluate SeededRandomPolicy or a checkpoint on maps/two_rooms.wad (20 seeds).

Reports exits, deaths, timeouts, kills, damage_taken, and door opening/passability,
and retains representative replays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import CONTRACT_VERSION, make_env  # noqa: E402
from doom_env.spaces import FRAME_SKIP  # noqa: E402
from submission.agent import DoomAgent  # noqa: E402
from training.eval_random import SeededRandomPolicy  # noqa: E402

WAD_PATH = ROOT / "maps" / "two_rooms.wad"
EPISODE_TICKS = 2100
ENV_SEEDS = list(range(20))
POLICY_SEEDS = [1000 + i for i in range(20)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_eval(weights_path: Path | None, out_dir: Path, overwrite: bool = False) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(WAD_PATH, "MAP01", episode_ticks=EPISODE_TICKS)
    policy = (
        SeededRandomPolicy(env.action_space.shape, seed=0)
        if weights_path is None
        else DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=weights_path, seed=0)
    )

    episodes: list[dict] = []
    try:
        for ep_idx, (env_seed, policy_seed) in enumerate(zip(ENV_SEEDS, POLICY_SEEDS)):
            if isinstance(policy, SeededRandomPolicy):
                policy.reset(seed=policy_seed)
            else:
                policy.seed = policy_seed
                policy.reset()

            rec_path = replay_dir / f"two_rooms_ep{ep_idx}_seed{policy_seed}.lmp"
            obs, _ = env.reset(seed=env_seed, options={"record_path": str(rec_path)})
            st0 = env.privileged_state()
            door_secs = [int(i) for i in np.where(st0["door_lock"] >= 0)[0]]

            decisions = 0
            total_reward = 0.0
            door_initiated = False
            door_passable = False
            terminated = truncated = False

            while not (terminated or truncated):
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                decisions += 1
                st = env.privileged_state()
                for ds in door_secs:
                    clr = float(st["sector_ceiling"][ds]) - float(st["sector_floor"][ds])
                    if clr > 0.0:
                        door_initiated = True
                    if clr >= 56.0:
                        door_passable = True

            st = env.privileged_state()
            outcome = "exit" if (terminated and info.get("exit_reached", False)) else ("death" if terminated else "timeout")
            episodes.append(
                {
                    "episode_idx": ep_idx,
                    "env_seed": env_seed,
                    "policy_seed": policy_seed,
                    "decisions": decisions,
                    "nominal_action_ticks": decisions * FRAME_SKIP,
                    "outcome": outcome,
                    "exit_reached": bool(info.get("exit_reached", False)),
                    "dead": bool(st["dead"]),
                    "timeout": bool(st["timeout"]),
                    "kills": int(st["kills"]),
                    "damage_taken": float(st["damage_taken"]),
                    "door_initiated": door_initiated,
                    "door_passable": door_passable,
                    "env_reward": total_reward,
                }
            )
    finally:
        env.close()

    # Keep up to 1 representative of each observed outcome (exit, death, timeout)
    kept_outcomes: set[str] = set()
    keep_paths: set[Path] = set()
    kept_meta: list[dict] = []
    for ep in episodes:
        p = replay_dir / f"two_rooms_ep{ep['episode_idx']}_seed{ep['policy_seed']}.lmp"
        if ep["outcome"] not in kept_outcomes and p.exists():
            kept_outcomes.add(ep["outcome"])
            keep_paths.add(p)
            kept_meta.append({"outcome": ep["outcome"], "file": p.name, "decisions": ep["decisions"], "kills": ep["kills"], "damage_taken": ep["damage_taken"], "door_passable": ep["door_passable"]})
    for ep in episodes:
        p = replay_dir / f"two_rooms_ep{ep['episode_idx']}_seed{ep['policy_seed']}.lmp"
        if p.exists() and p not in keep_paths:
            p.unlink()
    (replay_dir / "replays.json").write_text(json.dumps(kept_meta, indent=2) + "\n")

    with (out_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)

    exits = [e for e in episodes if e["outcome"] == "exit"]
    summary = {
        "contract_version": CONTRACT_VERSION,
        "wad": "maps/two_rooms.wad",
        "wad_sha256": sha256_file(WAD_PATH),
        "policy": str(weights_path) if weights_path else "SeededRandomPolicy",
        "checkpoint_sha256": sha256_file(weights_path) if weights_path else None,
        "episode_ticks_cap": EPISODE_TICKS,
        "env_seeds": ENV_SEEDS,
        "policy_seeds": POLICY_SEEDS,
        "episodes": len(episodes),
        "exits": len(exits),
        "deaths": sum(1 for e in episodes if e["outcome"] == "death"),
        "timeouts": sum(1 for e in episodes if e["outcome"] == "timeout"),
        "door_initiated_count": sum(1 for e in episodes if e["door_initiated"]),
        "door_passable_count": sum(1 for e in episodes if e["door_passable"]),
        "kills_mean": float(np.mean([e["kills"] for e in episodes])),
        "kills_nonzero_episodes": sum(1 for e in episodes if e["kills"] > 0),
        "damage_taken_mean": float(np.mean([e["damage_taken"] for e in episodes])),
        "decisions_mean": float(np.mean([e["decisions"] for e in episodes])),
        "exit_decisions_mean": float(np.mean([e["decisions"] for e in exits])) if exits else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_eval(args.weights, args.out_dir, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
