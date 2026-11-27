"""Validate procedural maps: D4 split separation, spawns, connectivity, WAD load, and engine checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from doom_env import make_env
from training.procedural.generate import ROOT, bfs_distances, d4_canonical
from training.progression.generate_progression import DOOR_SPEC


def verify() -> list[dict]:
    manifest = json.loads((ROOT / "training/procedural/manifest.json").read_text())
    maps = manifest["maps"]

    seen_d4: dict[tuple[str, ...], str] = {}
    rows: list[dict] = []
    rep_checked: set[str] = set()

    for m in maps:
        path = ROOT / m["wad_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"], f"Hash mismatch: {m['name']}"
        grid = m["grid"]

        # 1. D4 uniqueness across all maps and splits
        sig = d4_canonical(grid)
        if sig in seen_d4:
            raise AssertionError(f"Duplicate D4 geometry: {m['name']} matches {seen_d4[sig]}")
        seen_d4[sig] = m["name"]

        # 2. Player, exit, and thing placement checks (no walls, no overlaps)
        p_cells = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "P"]
        e_cells = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "E"]
        assert len(p_cells) == 1 and len(e_cells) == 1, f"Invalid P/E count in {m['name']}"
        pr, pc = p_cells[0]
        er, ec = e_cells[0]

        occupied = {(pr, pc), (er, ec)}
        for th in m.get("things", []):
            rc = (th["r"], th["c"])
            assert grid[rc[0]][rc[1]] == ".", f"Thing spawned on non-floor {grid[rc[0]][rc[1]]!r} in {m['name']}"
            assert rc not in occupied, f"Overlapping thing spawn at {rc} in {m['name']}"
            occupied.add(rc)

        # 3. Exit connectivity & key-before-door ordering
        all_dist = bfs_distances(grid, (pr, pc))
        assert (er, ec) in all_dist, f"Exit unreachable in {m['name']}"

        if m["task"] == "unlocked_door":
            no_door = bfs_distances(grid, (pr, pc), blocked_chars={"#", "D"})
            assert (er, ec) not in no_door, f"Unlocked door bypassed in {m['name']}"

        if m["task"] == "progression":
            door_chars = [ch for row in grid for ch in row if ch in DOOR_SPEC]
            assert len(door_chars) == 1, f"Expected 1 locked door in {m['name']}"
            dch = door_chars[0]
            kch = DOOR_SPEC[dch]["key_char"]
            k_cells = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == kch]
            assert len(k_cells) == 1, f"Expected 1 matching key {kch!r} in {m['name']}"
            before_door = bfs_distances(grid, (pr, pc), blocked_chars={"#", dch})
            assert (er, ec) not in before_door, f"Locked door bypassed in {m['name']}"
            assert k_cells[0] in before_door, f"Key unreachable before locked door in {m['name']}"

        # 4. Engine load + contract check + representative door/exit check
        env = make_env(path, m["map_name"], episode_ticks=2100, render_buffers=False)
        try:
            obs, _ = env.reset(seed=0)
            assert env.observation_space.contains(obs), f"Obs space mismatch in {m['name']}"
            assert env.action_space.shape == (9,)
            st = env.privileged_state()
            if m["task"] == "unlocked_door" or m.get("has_door", False):
                assert np.any(st["door_lock"] == 0), f"Missing unlocked door sector in {m['name']}"
            if m["task"] == "progression":
                assert np.any(st["door_lock"] > 0), f"Missing locked door sector in {m['name']}"
            if m["task"] == "combat":
                count = sum(o.name in ("Zombieman", "DoomImp") for o in env._game.get_state().objects)
                assert count == m["enemies"], f"Enemy count mismatch in {m['name']}: {count} != {m['enemies']}"

            # Representative engine check: verify action stepping works cleanly
            if m["task"] not in rep_checked:
                act = np.zeros(9, dtype=np.int8)
                act[0] = 1
                obs2, _, _, _, _ = env.step(act)
                assert env.observation_space.contains(obs2)
                rep_checked.add(m["task"])
        finally:
            env.close()

        rows.append(
            {
                "name": m["name"],
                "split": m["split"],
                "task": m["task"],
                "seed": m["seed"],
                "sha256": m["sha256"],
                "verified": True,
            }
        )

    out = ROOT / "training/results/procedural_map_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Verified {len(rows)} procedural maps; D4 uniqueness, spawns, connectivity, and WAD loads passed.")
    return rows


if __name__ == "__main__":
    verify()
