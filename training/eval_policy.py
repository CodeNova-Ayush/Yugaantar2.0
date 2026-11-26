"""Evaluate a trained (or untrained) checkpoint through DoomAgent on any manifest splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import CONTRACT_VERSION, make_env  # noqa: E402
from doom_env.spaces import FRAME_SKIP  # noqa: E402
from submission.agent import DoomAgent  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "curriculum" / "manifest.json"
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


def classify_outcome(terminated: bool, truncated: bool, info: dict) -> str:
    if terminated and info.get("exit_reached", False):
        return "exit"
    if terminated:
        return "death"
    if truncated:
        return "timeout"
    raise RuntimeError(f"Episode ended without terminated or truncated: {terminated=}, {truncated=}")


def evaluate_layout(
    weights_path: Path,
    map_entry: dict,
    replay_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    wad_path = ROOT / map_entry["wad_path"]
    wad_sha256 = sha256_file(wad_path)
    assert wad_sha256 == map_entry["sha256"], f"Map SHA-256 mismatch for {wad_path}"

    env = make_env(wad_path, map_entry["map_name"], episode_ticks=EPISODE_TICKS)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=weights_path, seed=0)
    episodes: list[dict] = []
    act_latencies_ms: list[float] = []
    saved_success = False
    saved_fail = False

    try:
        for ep_idx, (env_seed, policy_seed) in enumerate(zip(ENV_SEEDS, POLICY_SEEDS)):
            agent.seed = policy_seed
            agent.reset()

            rec_path = None
            if replay_dir is not None:
                replay_dir.mkdir(parents=True, exist_ok=True)
                rec_path = replay_dir / f"{map_entry['name']}_ep{ep_idx}_seed{policy_seed}.lmp"

            opts = {"record_path": str(rec_path)} if rec_path is not None else None
            obs, info = env.reset(seed=env_seed, options=opts)
            assert env.observation_space.contains(obs)

            decisions = 0
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                t0 = time.perf_counter()
                action = agent.act(obs)
                act_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
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

    # Finalize replay retention only after env.close() flushes all .lmp recordings.
    if replay_dir is not None:
        retained: list[dict] = []
        keep_paths: set[Path] = set()
        for ep in episodes:
            p = replay_dir / f"{map_entry['name']}_ep{ep['episode_idx']}_seed{ep['policy_seed']}.lmp"
            if ep["outcome"] == "exit" and not saved_success and p.exists():
                saved_success = True
                keep_paths.add(p)
                retained.append({"outcome": ep["outcome"], "file": p.name, "decisions": ep["decisions"], "policy_seed": ep["policy_seed"]})
            elif ep["outcome"] != "exit" and not saved_fail and p.exists():
                saved_fail = True
                keep_paths.add(p)
                retained.append({"outcome": ep["outcome"], "file": p.name, "decisions": ep["decisions"], "policy_seed": ep["policy_seed"]})
        for ep in episodes:
            p = replay_dir / f"{map_entry['name']}_ep{ep['episode_idx']}_seed{ep['policy_seed']}.lmp"
            if p.exists() and p not in keep_paths:
                p.unlink()
        index_path = replay_dir / f"{map_entry['name']}_replays.json"
        index_path.write_text(json.dumps(retained, indent=2) + "\n")

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
        "act_latency_ms_mean": float(np.mean(act_latencies_ms)) if act_latencies_ms else None,
        "act_latency_ms_p50": float(np.percentile(act_latencies_ms, 50)) if act_latencies_ms else None,
        "act_latency_ms_p99": float(np.percentile(act_latencies_ms, 99)) if act_latencies_ms else None,
    }
    return episodes, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to evaluate (pass 'test' explicitly when frozen).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--save-replays-dir", type=Path, default=None)
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
    selected_maps = [m for m in manifest["maps"] if m["split"] in set(args.splits)]
    if not selected_maps:
        raise ValueError(f"No maps matched splits={args.splits} in {args.manifest}")

    all_episodes: list[dict] = []
    summaries: list[dict] = []
    for map_entry in selected_maps:
        eps, summ = evaluate_layout(args.weights, map_entry, replay_dir=args.save_replays_dir)
        all_episodes.extend(eps)
        summaries.append(summ)

    with (args.out_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_episodes[0].keys()))
        writer.writeheader()
        writer.writerows(all_episodes)

    payload = {
        "contract_version": CONTRACT_VERSION,
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "manifest": str(args.manifest),
        "splits": args.splits,
        "command": " ".join(sys.argv),
        "git": git_revision(),
        "versions": dependency_versions(),
        "config": {
            "episode_ticks_cap": EPISODE_TICKS,
            "frame_skip": FRAME_SKIP,
            "env_seeds": ENV_SEEDS,
            "policy_seeds": POLICY_SEEDS,
            "evaluator": "DoomAgent (stochastic Bernoulli sampling with per-episode torch.Generator)",
        },
        "summaries": summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
