"""Comparable evaluation across the four procedural task families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from submission.agent import DoomAgent
from training.eval_random import SeededRandomPolicy
from training.task_rewards import TaskRewardTracker

ROOT = Path(__file__).resolve().parents[2]


def evaluate(
    weights: Path | None,
    manifest: Path = ROOT / "training/procedural/manifest.json",
    split: str = "val",
    episodes: int = 20,
    seed_offset: int = 1000,
) -> dict:
    torch.set_num_threads(1)
    maps = [m for m in json.loads(Path(manifest).read_text())["maps"] if m["split"] == split]
    rows = []
    for m in maps:
        path = ROOT / m["wad_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != m["sha256"]:
            raise ValueError("Map changed: " + m["name"])
        env = make_env(path, m["map_name"], episode_ticks=2100, render_buffers=False)
        try:
            agent = (
                DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=Path(weights))
                if weights
                else SeededRandomPolicy(env.action_space.shape, seed=0)
            )
            for seed in range(episodes):
                if weights:
                    agent.seed = seed_offset + seed
                    agent.reset()
                else:
                    agent.reset(seed=seed_offset + seed)
                obs, _ = env.reset(seed=seed)
                tracker = TaskRewardTracker(env.privileged_state())
                done = False
                steps = 0
                key = False
                passable = False
                unlocked_passable = False
                while not done:
                    obs, r, term, trunc, info = env.step(agent.act(obs))
                    steps += 1
                    done = term or trunc
                    st = env.privileged_state()
                    tracker.step(st)
                    key |= bool(np.any(st["keys"]))
                    locks = st["door_lock"]
                    if locks is not None:
                        clearance = st["sector_ceiling"] - st["sector_floor"]
                        if key:
                            passable |= bool(np.any((locks > 0) & (clearance >= 56)))
                        unlocked_passable |= bool(np.any((locks == 0) & (clearance >= 56)))
                rows.append(
                    {
                        "layout": m["name"],
                        "task": m["task"],
                        "env_seed": seed,
                        "policy_seed": seed_offset + seed,
                        "exit": bool(info.get("exit_reached")),
                        "death": st["dead"],
                        "timeout": st["timeout"],
                        "kills": st["kills"],
                        "damage": st["damage_taken"],
                        "decisions": steps,
                        "key": key,
                        "passable_after_key": passable,
                        "unlocked_door_passable": unlocked_passable,
                        "unique_cells": len(tracker.visited),
                    }
                )
        finally:
            env.close()

    tasks = {}
    for task in sorted({m["task"] for m in maps}):
        rs = [r for r in rows if r["task"] == task]
        n = len(rs)
        tasks[task] = {
            "episodes": n,
            **{
                k: sum(r[k] for r in rs)
                for k in ["exit", "death", "timeout", "kills", "key", "passable_after_key", "unlocked_door_passable"]
            },
            "exit_rate": sum(r["exit"] for r in rs) / n,
            "mean_decisions": sum(r["decisions"] for r in rs) / n,
        }
    return {
        "weights": str(weights) if weights else None,
        "sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest() if weights else None,
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "split": split,
        "episode_ticks": 2100,
        "tasks": tasks,
        "balanced_exit_rate": float(np.mean([t["exit_rate"] for t in tasks.values()])),
        "episodes": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path)
    p.add_argument("--manifest", type=Path, default=ROOT / "training/procedural/manifest.json")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = evaluate(a.weights, a.manifest, a.split, a.episodes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["tasks"], indent=2))
