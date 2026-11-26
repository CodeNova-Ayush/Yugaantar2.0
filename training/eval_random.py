"""Reproducible standalone random-agent evaluation harness.

Uses the official doom_env wrapper (observations, 9-button MultiBinary action
space, 4-tick decisions) and a standalone SeededRandomPolicy that preserves the
starter's independent 50/50 Bernoulli(0.5)^9 distribution without importing the
evolving submission agent.
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

from doom_env import CONTRACT_VERSION, make_env
from doom_env.spaces import FRAME_SKIP

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "maps" / "manifest.json"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "random_baseline_v2"

EPISODE_TICKS = 2100
ENV_SEEDS = list(range(20))
POLICY_SEEDS = [1000 + i for i in range(20)]


class SeededRandomPolicy:
    """Standalone uniform Bernoulli(0.5)^9 policy matching the starter baseline."""

    def __init__(self, action_shape: tuple[int, ...], seed: int = 0):
        self.action_shape = action_shape
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation: dict) -> np.ndarray:
        del observation  # uniform random policy ignores observations
        return self.rng.integers(0, 2, size=self.action_shape, dtype=np.int8)


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
    pkgs = ["doom-env", "vizdoom", "gymnasium", "numpy", "torch"]
    out = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "not-installed"
    return out


def classify_outcome(terminated: bool, truncated: bool, info: dict) -> str:
    """Distinguish exit, death, and timeout using official wrapper signals."""
    if terminated and info.get("exit_reached", False):
        return "exit"
    if terminated:
        return "death"
    if truncated:
        return "timeout"
    raise RuntimeError(f"Episode ended without terminated or truncated: {terminated=}, {truncated=}")


def evaluate_layout(map_entry: dict) -> tuple[list[dict], dict]:
    wad_path = ROOT / map_entry["wad_path"]
    wad_sha256 = sha256_file(wad_path)
    assert wad_sha256 == map_entry["sha256"], (
        f"Hash mismatch for {wad_path}: {wad_sha256} != {map_entry['sha256']}"
    )

    env = make_env(wad_path, map_entry["map_name"], episode_ticks=EPISODE_TICKS)
    policy = SeededRandomPolicy(env.action_space.shape, seed=0)
    episodes: list[dict] = []

    try:
        for ep_idx, (env_seed, policy_seed) in enumerate(zip(ENV_SEEDS, POLICY_SEEDS)):
            obs, info = env.reset(seed=env_seed)
            assert env.observation_space.contains(obs)
            policy.reset(seed=policy_seed)

            decisions = 0
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                decisions += 1

            outcome = classify_outcome(terminated, truncated, info)
            episodes.append(
                {
                    "layout": map_entry["name"],
                    "split": map_entry["split"],
                    "wad_path": map_entry["wad_path"],
                    "wad_sha256": wad_sha256,
                    "episode_idx": ep_idx,
                    "env_seed": env_seed,
                    "policy_seed": policy_seed,
                    "episode_ticks_cap": EPISODE_TICKS,
                    "decisions": decisions,
                    "nominal_action_ticks": decisions * FRAME_SKIP,
                    "raw_engine_tic": int(info.get("tic", 0)),
                    "outcome": outcome,
                    "exit_reached": bool(info.get("exit_reached", False)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "env_reward": total_reward,
                }
            )
    finally:
        env.close()

    n = len(episodes)
    exits = [e for e in episodes if e["outcome"] == "exit"]
    deaths = sum(1 for e in episodes if e["outcome"] == "death")
    timeouts = sum(1 for e in episodes if e["outcome"] == "timeout")
    lengths = [e["decisions"] for e in episodes]
    exit_lengths = [e["decisions"] for e in exits]

    summary = {
        "layout": map_entry["name"],
        "split": map_entry["split"],
        "wad_path": map_entry["wad_path"],
        "wad_sha256": wad_sha256,
        "map_name": map_entry["map_name"],
        "episode_ticks_cap": EPISODE_TICKS,
        "episodes": n,
        "exits": len(exits),
        "exit_rate": len(exits) / n if n else 0.0,
        "deaths": deaths,
        "timeouts": timeouts,
        "decisions_mean": float(np.mean(lengths)),
        "decisions_std": float(np.std(lengths)),
        "decisions_min": int(np.min(lengths)),
        "decisions_max": int(np.max(lengths)),
        "exit_decisions_mean": float(np.mean(exit_lengths)) if exit_lengths else None,
        "exit_decisions_std": float(np.std(exit_lengths)) if exit_lengths else None,
    }
    return episodes, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    maps = manifest["maps"]
    if args.splits is not None:
        maps = [m for m in maps if m["split"] in set(args.splits)]
    else:
        maps = [m for m in maps if m["split"] != "test"]

    all_episodes: list[dict] = []
    summaries: list[dict] = []
    for map_entry in maps:
        eps, summ = evaluate_layout(map_entry)
        all_episodes.extend(eps)
        summaries.append(summ)

    csv_path = out_dir / "episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_episodes[0].keys()))
        writer.writeheader()
        writer.writerows(all_episodes)

    summary_payload = {
        "contract_version": CONTRACT_VERSION,
        "policy": "SeededRandomPolicy (independent Bernoulli(0.5)^9)",
        "command": " ".join(sys.argv),
        "git": git_revision(),
        "versions": dependency_versions(),
        "config": {
            "episode_ticks_cap": EPISODE_TICKS,
            "frame_skip": FRAME_SKIP,
            "env_seeds": ENV_SEEDS,
            "policy_seeds": POLICY_SEEDS,
        },
        "duration_note": (
            "decisions is the primary duration metric; nominal_action_ticks = decisions * 4; "
            "raw_engine_tic is diagnostic only (ViZDoom resets get_episode_time() to 0 on Exit_Normal)."
        ),
        "seed_effect_note": (
            "On these static, enemy-free navigation maps, env_seed has no visible "
            "effect on spawn position, geometry, or physics; all episode-to-episode "
            "variation comes from policy_seed."
        ),
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
