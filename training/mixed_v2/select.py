"""Evaluate all saved checkpoints on mixed_v2 validation; select by predeclared equal-task exit rate.

Predeclared selection rule:
  Maximize balanced_exit_rate = mean(exit_rate over {navigation, unlocked_door, progression, combat_rooms}).
  Tie-breakers: combat_rooms kills per episode, fewer mean decisions, earlier checkpoint.
Test data is never read by this script. Existing evaluation files are hash checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.mixed_v2.evaluate import ROOT, evaluate


def select(run: Path, episodes: int = 20) -> dict:
    manifest = ROOT / "training/mixed_v2/manifest.json"
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
            assert all(
                t["episodes"]
                == episodes
                * sum(m["task"] == name and m["split"] == "val" for m in json.loads(manifest.read_text())["maps"])
                for name, t in r["tasks"].items()
            )
        else:
            r = evaluate(w, manifest, split="val", episodes=episodes)
            out.write_text(json.dumps(r, indent=2) + "\n")
        t = r["tasks"]
        combat = t["combat_rooms"]
        mean = sum(e["decisions"] for e in r["episodes"]) / len(r["episodes"])
        rank = (r["balanced_exit_rate"], combat["kills"] / combat["episodes"], -mean, -n)
        results.append({"decisions": n, "checkpoint": str(w), "sha256": r["sha256"], "rank": rank, "tasks": t})
        print(n, {k: round(v["exit_rate"], 3) for k, v in t.items()}, "bal=", round(r["balanced_exit_rate"], 4), flush=True)
    if not results:
        raise ValueError("No checkpoints")
    selected = max(results, key=lambda r: r["rank"])
    payload = {
        "selection_rule": "Equal-task mean validation exit rate over 4 mixed_v2 tasks; ties: combat_rooms kills/episode, fewer decisions, earlier checkpoint",
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
