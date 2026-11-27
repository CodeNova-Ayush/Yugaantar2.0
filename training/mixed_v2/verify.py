"""Verify Curriculum v2 (mixed_v2) D4 split separation, loading, doors, enemies, and exit reachability."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from doom_env import make_env
from training.mixed_v2.generate import ROOT


def _d4_canonical(grid: list[str]) -> tuple[str, ...]:
    """Canonical D4 signature of walkable geometry (# vs non-#)."""
    norm = ["".join("#" if ch == "#" else "." for ch in row) for row in grid]
    variants = []
    cur = norm
    for _ in range(4):
        variants.append(tuple(cur))
        variants.append(tuple(row[::-1] for row in cur))
        cur = ["".join(cur[len(cur) - 1 - r][c] for r in range(len(cur))) for c in range(len(cur[0]))]
    return min(variants)


def verify() -> list[dict]:
    manifest = json.loads((ROOT / "training/mixed_v2/manifest.json").read_text())
    maps = manifest["maps"]

    # 1. D4 separation for unlocked_door across splits
    split_sigs: dict[str, set[tuple[str, ...]]] = {"train": set(), "val": set(), "test": set()}
    for m in maps:
        if m["task"] == "unlocked_door":
            sig = _d4_canonical(m["grid"])
            for other_split, sigs in split_sigs.items():
                if other_split != m["split"] and sig in sigs:
                    raise AssertionError(f"D4 geometry leak for {m['name']} into {other_split}")
            split_sigs[m["split"]].add(sig)

    rows = []
    for m in maps:
        if m["task"] not in ("unlocked_door", "combat_rooms"):
            continue
        path = ROOT / m["wad_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"], f"Hash mismatch: {m['name']}"

        # 2. Geometric reachability from P to E
        grid = m["grid"]
        start = next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "P")
        seen = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#" and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        assert any(grid[r][c] == "E" for r, c in seen), f"Unreachable exit in {m['name']}"

        # 3. Engine load, observation space, doors, and enemies
        env = make_env(path, m["map_name"], episode_ticks=2100, render_buffers=False)
        try:
            obs, _ = env.reset(seed=0)
            assert env.observation_space.contains(obs), f"Invalid obs in {m['name']}"
            st = env.privileged_state()
            if m["task"] == "unlocked_door" or m.get("has_door", False):
                assert np.any(st["door_lock"] == 0), f"Expected unlocked door in {m['name']}"
            if m["task"] == "combat_rooms":
                count = sum(o.name in ("Zombieman", "DoomImp") for o in env._game.get_state().objects)
                assert count == m["enemies"], f"Enemy count mismatch in {m['name']}: {count} != {m['enemies']}"
            rows.append(
                {
                    "layout": m["name"],
                    "split": m["split"],
                    "task": m["task"],
                    "enemies": m.get("enemies", 0),
                    "has_door": m["task"] == "unlocked_door" or m.get("has_door", False),
                    "geometric_exit_reachable": True,
                }
            )
        finally:
            env.close()

    out = ROOT / "training/results/mixed_v2_map_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Verified {len(rows)} unlocked_door and combat_rooms layouts; D4 split separation passed.")
    return rows


if __name__ == "__main__":
    verify()
