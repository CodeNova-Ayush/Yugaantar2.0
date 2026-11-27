"""Predeclared validation-only checkpoint selection for procedural runs.

Selection rule:
  Maximize balanced_exit_rate = mean(exit_rate over {navigation, unlocked_door, progression, combat}).
  Tie-breakers: combat kills per episode, fewer mean decisions, earlier checkpoint.
Test data is never read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.procedural.evaluate import ROOT, evaluate


def select(run: Path, episodes: int = 20) -> dict:
    manifest = ROOT / "training/procedural/manifest.json"
    results = []
    if not (run / "summary.json").exists():
        raise ValueError("Training must finish before final checkpoint selection")
    for w in sorted((run / "checkpoints").glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        n = int(w.stem.split("_")[1])
        out = run / f"validation_{n}.json"
        if out.exists():
            r = json.loads(out.read_text())
            assert r["sha256"] == hashlib.sha256(w.read_bytes()).hexdigest()
            assert r["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
            expected = {(m["name"], seed, 1000 + seed) for m in json.loads(manifest.read_text())["maps"] if m["split"] == "val" for seed in range(episodes)}
            actual = {(e["layout"], e["env_seed"], e["policy_seed"]) for e in r["episodes"]}
            if r.get("split") != "val" or actual != expected or len(r["episodes"]) != len(expected):
                raise ValueError("Cached evaluation split/seeds/episode count mismatch; use a separate output directory")
        else:
            r = evaluate(w, manifest, split="val", episodes=episodes)
            out.write_text(json.dumps(r, indent=2) + "\n")
        t = r["tasks"]
        combat = t["combat"]
        mean = sum(e["decisions"] for e in r["episodes"]) / len(r["episodes"])
        rank = (r["balanced_exit_rate"], combat["kills"] / combat["episodes"], -mean, -n)
        results.append({"decisions": n, "checkpoint": str(w), "sha256": r["sha256"], "rank": rank, "tasks": t})
        print(n, {k: round(v["exit_rate"], 3) for k, v in t.items()}, "bal=", round(r["balanced_exit_rate"], 4), flush=True)
    if not results:
        raise ValueError("No checkpoints")
    selected = max(results, key=lambda r: r["rank"])
    payload = {
        "selection_rule": "Equal-task mean validation exit rate over 4 procedural tasks; ties: combat kills/episode, fewer decisions, earlier checkpoint",
        "episodes_per_layout": episodes,
        "selected": selected,
        "curve": results,
    }
    (run / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    return selected


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--episodes", type=int, default=20)
    a = p.parse_args()
    print(select(a.run, a.episodes))
