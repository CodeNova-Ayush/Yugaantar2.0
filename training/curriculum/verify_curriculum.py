"""Verify curriculum split hygiene (no D4-equivalent geometry across splits) and exit reachability."""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env  # noqa: E402
from doom_env.spaces import FRAME_SKIP  # noqa: E402
from training.generate_maps import CELL_SIZE  # noqa: E402

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
TURN_L = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
TURN_R = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)


def canonical_d4_signature(grid: list[str]) -> tuple[str, ...]:
    """Return canonical D4 (rotations + reflections) signature of walkable mask (ignoring P and E)."""
    arr = np.array([[0 if ch == "#" else 1 for ch in row] for row in grid])
    variants = []
    for k in range(4):
        rot = np.rot90(arr, k)
        for mat in (rot, np.fliplr(rot)):
            # Trim bounding box of non-wall cells
            ys, xs = np.where(mat > 0)
            cropped = mat[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            variants.append(tuple("".join(str(v) for v in row) for row in cropped))
    return min(variants)


def check_split_hygiene(maps: list[dict]) -> None:
    sigs_by_split: dict[str, set[tuple[str, ...]]] = {}
    for m in maps:
        sig = canonical_d4_signature(m["grid"])
        sigs_by_split.setdefault(m["split"], set()).add(sig)
    splits = list(sigs_by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = sigs_by_split[splits[i]] & sigs_by_split[splits[j]]
            assert not overlap, f"D4 geometry overlap between {splits[i]} and {splits[j]}: {overlap}"


def bfs_cell_path(grid: list[str]) -> list[tuple[int, int]]:
    h, w = len(grid), len(grid[0])
    start = None
    for r in range(h):
        for c in range(w):
            if grid[r][c] == "P":
                start = (r, c)
    assert start is not None
    q = deque([(start, [start])])
    seen = {start}
    while q:
        (r, c), path = q.popleft()
        if grid[r][c] == "E":
            return path
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != "#" and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), path + [(nr, nc)]))
    raise AssertionError("No walkable path from P to E")


def verify_map_exit(m: dict) -> dict:
    grid = m["grid"]
    h = len(grid)
    path = bfs_cell_path(grid)
    waypoints = [((c + 0.5) * CELL_SIZE, (h - 1 - r + 0.5) * CELL_SIZE) for r, c in path]

    env = make_env(ROOT / m["wad_path"], m["map_name"], episode_ticks=2100, render_buffers=False)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        decisions = 0
        wp_idx = 1
        term = trunc = False
        while not (term or trunc) and decisions < 400:
            st = env.privileged_state()
            tx, ty = waypoints[min(wp_idx, len(waypoints) - 1)]
            dx, dy = tx - st["x"], ty - st["y"]
            if math.hypot(dx, dy) < 20.0 and wp_idx < len(waypoints) - 1:
                wp_idx += 1
                tx, ty = waypoints[wp_idx]
                dx, dy = tx - st["x"], ty - st["y"]
            target_deg = math.degrees(math.atan2(dy, dx)) % 360.0
            diff = (target_deg - st["angle"] + 180.0) % 360.0 - 180.0
            if diff > 12.0:
                act = TURN_L
            elif diff < -12.0:
                act = TURN_R
            else:
                act = FWD
            obs, _, term, trunc, info = env.step(act)
            decisions += 1
        st = env.privileged_state()
        assert term and not trunc and info["exit_reached"] and not st["dead"], (
            f"Map {m['name']} exit failed: term={term}, trunc={trunc}, info={info}"
        )
        return {
            "name": m["name"],
            "split": m["split"],
            "decisions": decisions,
            "nominal_action_ticks": decisions * FRAME_SKIP,
            "exit_reached": True,
        }
    finally:
        env.close()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    maps = manifest["maps"]
    check_split_hygiene(maps)
    results = [verify_map_exit(m) for m in maps]
    print(json.dumps({"verified_count": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
