"""Aggregate Stage 3 comparison from existing episode CSVs (no new training)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"

POLICY_CSVS = {
    "random": RESULTS_DIR / "curriculum_eval_random" / "episodes.csv",
    "untrained": RESULTS_DIR / "curriculum_eval_untrained" / "episodes.csv",
    "ppo_s42": RESULTS_DIR / "curriculum_eval_s42" / "episodes.csv",
    "ppo_s43": RESULTS_DIR / "curriculum_eval_s43" / "episodes.csv",
    "ppo_s44": RESULTS_DIR / "curriculum_eval_s44" / "episodes.csv",
}


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize_group(rows: list[dict]) -> dict:
    n = len(rows)
    exits = [r for r in rows if r["outcome"] == "exit"]
    timeouts = sum(1 for r in rows if r["outcome"] == "timeout")
    deaths = sum(1 for r in rows if r["outcome"] == "death")
    decs = [int(r["decisions"]) for r in rows]
    exit_decs = [int(r["decisions"]) for r in exits]
    return {
        "episodes": n,
        "exits": len(exits),
        "exit_rate": len(exits) / n if n else 0.0,
        "deaths": deaths,
        "timeouts": timeouts,
        "decisions_mean": float(np.mean(decs)),
        "decisions_std": float(np.std(decs)),
        "exit_decisions_mean": float(np.mean(exit_decs)) if exit_decs else None,
        "exit_decisions_std": float(np.std(exit_decs)) if exit_decs else None,
    }


def main() -> None:
    data = {name: load_rows(path) for name, path in POLICY_CSVS.items()}

    # 1. Split-level summaries per policy
    split_summary: dict[str, dict] = {}
    for name, rows in data.items():
        split_summary[name] = {}
        for split in ("train", "val", "test"):
            split_summary[name][split] = summarize_group([r for r in rows if r["split"] == split])

    # 2. Variation across the 3 PPO training seeds (42, 43, 44)
    seed_variation: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        rates = [split_summary[s][split]["exit_rate"] for s in ("ppo_s42", "ppo_s43", "ppo_s44")]
        decs = [split_summary[s][split]["decisions_mean"] for s in ("ppo_s42", "ppo_s43", "ppo_s44")]
        exit_decs = [
            split_summary[s][split]["exit_decisions_mean"]
            for s in ("ppo_s42", "ppo_s43", "ppo_s44")
            if split_summary[s][split]["exit_decisions_mean"] is not None
        ]
        seed_variation[split] = {
            "exit_rate_mean": float(np.mean(rates)),
            "exit_rate_std": float(np.std(rates)),
            "exit_rate_min": float(np.min(rates)),
            "exit_rate_max": float(np.max(rates)),
            "decisions_mean_across_seeds": float(np.mean(decs)),
            "decisions_std_across_seeds": float(np.std(decs)),
            "exit_decisions_mean_across_seeds": float(np.mean(exit_decs)) if exit_decs else None,
            "exit_decisions_std_across_seeds": float(np.std(exit_decs)) if exit_decs else None,
        }

    # 3. Per-layout breakdown
    layouts_ordered = []
    seen = set()
    for r in data["random"]:
        if r["layout"] not in seen:
            seen.add(r["layout"])
            layouts_ordered.append((r["layout"], r["split"]))

    per_layout: list[dict] = []
    for layout, split in layouts_ordered:
        entry = {"layout": layout, "split": split, "policies": {}}
        for name, rows in data.items():
            entry["policies"][name] = summarize_group([r for r in rows if r["layout"] == layout])
        per_layout.append(entry)

    payload = {
        "note": (
            "Aggregated from existing Stage 3 episode CSVs. Training duration (51,200 vs 20,480) "
            "and map diversity (12 vs 1) both changed from Stage 2, so Stage 3 is not a controlled "
            "map-diversity ablation."
        ),
        "split_summary": split_summary,
        "ppo_seed_variation": seed_variation,
        "per_layout": per_layout,
    }
    (RESULTS_DIR / "stage3_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Markdown report
    lines = [
        "# Stage 3 Aggregate Comparison (from Existing Episode CSVs)",
        "",
        "## Split Summary",
        "",
        "| Policy | Split | Exits / Eps | Exit Rate | Timeouts | Mean Decisions | Success-Only Mean Decisions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in POLICY_CSVS:
        for split in ("train", "val", "test"):
            s = split_summary[name][split]
            ed = f"{s['exit_decisions_mean']:.1f}" if s["exit_decisions_mean"] is not None else "n/a"
            lines.append(
                f"| {name} | {split} | {s['exits']}/{s['episodes']} | {100*s['exit_rate']:.1f}% | "
                f"{s['timeouts']} | {s['decisions_mean']:.1f} | {ed} |"
            )

    lines += [
        "",
        "## PPO 3-Seed Variation (Seeds 42, 43, 44)",
        "",
        "| Split | Exit Rate Mean ± Std (Min–Max) | Mean Decisions (± Std) | Success-Only Decisions (± Std) |",
        "| --- | --- | --- | --- |",
    ]
    for split in ("train", "val", "test"):
        v = seed_variation[split]
        lines.append(
            f"| {split} | {100*v['exit_rate_mean']:.1f}% ± {100*v['exit_rate_std']:.1f}% "
            f"({100*v['exit_rate_min']:.1f}%–{100*v['exit_rate_max']:.1f}%) | "
            f"{v['decisions_mean_across_seeds']:.1f} ± {v['decisions_std_across_seeds']:.1f} | "
            f"{v['exit_decisions_mean_across_seeds']:.1f} ± {v['exit_decisions_std_across_seeds']:.1f} |"
        )

    lines += [
        "",
        "## Per-Layout Exit Rates & Success-Only Mean Decisions",
        "",
        "| Layout | Split | Random | Untrained | PPO s42 | PPO s43 | PPO s44 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in per_layout:
        cells = []
        for name in ("random", "untrained", "ppo_s42", "ppo_s43", "ppo_s44"):
            p = item["policies"][name]
            ed = f"{p['exit_decisions_mean']:.0f}" if p["exit_decisions_mean"] is not None else "-"
            cells.append(f"{p['exits']}/20 ({ed})")
        lines.append(f"| {item['layout']} | {item['split']} | " + " | ".join(cells) + " |")

    (RESULTS_DIR / "stage3_comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS_DIR / 'stage3_comparison.json'} and {RESULTS_DIR / 'stage3_comparison.md'}")


if __name__ == "__main__":
    main()
