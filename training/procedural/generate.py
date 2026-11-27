"""Reproducible procedural DOOM map generator (proc_v1).

Generates varied multi-room, corridor, junction, dead-end, loop, unlocked-door,
key-and-locked-door progression, and multi-room combat layouts from explicit seeds.
Validates connectivity, key-before-door solvability, door framing, and D4 split
uniqueness before accepting any map.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from doom_env.wad import Lump, write_wad
from training.generate_maps import CELL_SIZE
from training.progression.generate_progression import DOOR_SPEC, KEY_SPEC

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "training/procedural"
MAPS_DIR = OUT_DIR / "maps"
GENERATOR_VERSION = "proc_v1"

TASK_WEIGHTS = {
    "navigation": 0.20,
    "unlocked_door": 0.25,
    "progression": 0.30,
    "combat": 0.25,
}


def d4_canonical(grid: list[str]) -> tuple[str, ...]:
    """Canonical D4 signature of walkable geometry (# vs non-#)."""
    norm = ["".join("#" if ch == "#" else "." for ch in row) for row in grid]
    variants = []
    cur = norm
    for _ in range(4):
        variants.append(tuple(cur))
        variants.append(tuple(row[::-1] for row in cur))
        cur = ["".join(cur[len(cur) - 1 - r][c] for r in range(len(cur))) for c in range(len(cur[0]))]
    return min(variants)


def bfs_distances(
    grid: list[str],
    start: tuple[int, int],
    blocked_chars: set[str] | None = None,
) -> dict[tuple[int, int], int]:
    h, w = len(grid), len(grid[0])
    blocked = blocked_chars or {"#"}
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist and grid[nr][nc] not in blocked:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _door_properly_framed(grid: list[str], r: int, c: int) -> bool:
    """A door cell must sit between two opposite walls (#) with walkable cells on the other axis."""
    h, w = len(grid), len(grid[0])
    if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
        return False
    ns_walls = grid[r - 1][c] == "#" and grid[r + 1][c] == "#"
    ew_open = grid[r][c - 1] not in ("#", "D", "R", "B", "Y") and grid[r][c + 1] not in ("#", "D", "R", "B", "Y")
    ew_walls = grid[r][c - 1] == "#" and grid[r][c + 1] == "#"
    ns_open = grid[r - 1][c] not in ("#", "D", "R", "B", "Y") and grid[r + 1][c] not in ("#", "D", "R", "B", "Y")
    return (ns_walls and ew_open) or (ew_walls and ns_open)


def procedural_grid_to_udmf(
    grid: list[str],
    player_angle: int = 0,
    extra_things: list[tuple[int, int, int, int]] | None = None,
    cell_size: int = CELL_SIZE,
) -> str:
    """Convert an ASCII grid (#, ., P, E, D, R/B/Y, r/b/y) + extra_things into UDMF TEXTMAP."""
    h, w = len(grid), len(grid[0])
    if any(len(r) != w for r in grid):
        raise ValueError("All grid rows must have equal width")

    # Sector 0 = main floor, Sector 1 = exit (GATE3), Sectors 2+ = door sectors.
    door_sectors: dict[tuple[int, int], tuple[int, str]] = {}
    next_sec = 2
    for r in range(h):
        for c in range(w):
            ch = grid[r][c]
            if ch == "D" and (r, c) not in door_sectors:
                q = deque([(r, c)])
                door_sectors[(r, c)] = (next_sec, "D")
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == "D" and (nr, nc) not in door_sectors:
                            door_sectors[(nr, nc)] = (next_sec, "D")
                            q.append((nr, nc))
                next_sec += 1
            elif ch in DOOR_SPEC and (r, c) not in door_sectors:
                door_sectors[(r, c)] = (next_sec, ch)
                next_sec += 1

    def cell_sector(r: int, c: int) -> int | None:
        if r < 0 or r >= h or c < 0 or c >= w:
            return None
        ch = grid[r][c]
        if ch in (".", "P") or ch in KEY_SPEC:
            return 0
        if ch == "E":
            return 1
        if (r, c) in door_sectors:
            return door_sectors[(r, c)][0]
        return None

    vertices: list[tuple[int, int]] = []
    vertex_idx: dict[tuple[int, int], int] = {}

    def vid(x: int, y: int) -> int:
        pt = (x, y)
        if pt not in vertex_idx:
            vertex_idx[pt] = len(vertices)
            vertices.append(pt)
        return vertex_idx[pt]

    sidedefs: list[str] = []
    linedefs: list[str] = []

    def add_sidedef(sector: int, top: str = "-", middle: str = "-", bottom: str = "-") -> int:
        idx = len(sidedefs)
        sidedefs.append(
            f'sidedef {{ sector = {sector}; texturetop = "{top}"; '
            f'texturemiddle = "{middle}"; texturebottom = "{bottom}"; }}'
        )
        return idx

    def add_wall(x1: int, y1: int, x2: int, y2: int, sector: int, tex: str = "STARTAN2") -> None:
        s = add_sidedef(sector, middle=tex)
        v1, v2 = vid(x1, y1), vid(x2, y2)
        linedefs.append(
            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {s}; "
            f"blocking = true; dontpegbottom = true; }}"
        )

    def add_exit_line(x1: int, y1: int, x2: int, y2: int, front_sec: int) -> None:
        sf = add_sidedef(front_sec)
        sb = add_sidedef(1)
        v1, v2 = vid(x1, y1), vid(x2, y2)
        linedefs.append(
            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; "
            f"twosided = true; special = 243; playercross = true; }}"
        )

    def add_door_line(x1: int, y1: int, x2: int, y2: int, front_sec: int, door_sec: int, dch: str) -> None:
        v1, v2 = vid(x1, y1), vid(x2, y2)
        if dch == "D":
            sf = add_sidedef(front_sec, top="BIGDOOR2")
            sb = add_sidedef(door_sec)
            linedefs.append(
                f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; "
                f"twosided = true; special = 12; arg0 = 0; arg1 = 16; arg2 = 150; "
                f"playeruse = true; monsteruse = true; repeatspecial = true; }}"
            )
        else:
            spec = DOOR_SPEC[dch]
            sf = add_sidedef(front_sec, top=spec["texture"])
            sb = add_sidedef(door_sec)
            linedefs.append(
                f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; "
                f"twosided = true; special = 13; arg0 = 0; arg1 = 16; arg2 = 0; "
                f"arg3 = {spec['lock_arg']}; playeruse = true; monsteruse = false; repeatspecial = true; }}"
            )

    player_pos: tuple[float, float] | None = None
    key_things: list[tuple[float, float, int]] = []

    for r in range(h):
        for c in range(w):
            sec = cell_sector(r, c)
            if sec is None:
                continue
            x0, x1 = c * cell_size, (c + 1) * cell_size
            y0, y1 = (h - 1 - r) * cell_size, (h - r) * cell_size
            ch = grid[r][c]
            cx, cy = x0 + cell_size / 2.0, y0 + cell_size / 2.0
            if ch == "P":
                player_pos = (cx, cy)
            elif ch in KEY_SPEC:
                key_things.append((cx, cy, KEY_SPEC[ch]["thing_type"]))

            neighbors = [
                (r - 1, c, x0, y1, x1, y1),
                (r, c + 1, x1, y1, x1, y0),
                (r + 1, c, x1, y0, x0, y0),
                (r, c - 1, x0, y0, x0, y1),
            ]
            for nr, nc, ax, ay, bx, by in neighbors:
                nsec = cell_sector(nr, nc)
                if nsec is None:
                    tex = "DOORTRAK" if sec >= 2 else ("STONE2" if sec == 1 else "STARTAN2")
                    add_wall(ax, ay, bx, by, sec, tex=tex)
                elif sec != 1 and nsec == 1:
                    add_exit_line(ax, ay, bx, by, sec)
                elif sec < 2 and nsec >= 2:
                    _, dch = door_sectors[(nr, nc)]
                    add_door_line(ax, ay, bx, by, sec, nsec, dch)

    if player_pos is None:
        raise ValueError("Grid must contain a 'P' player start")

    lines = ['namespace = "zdoom";']
    for vx, vy in vertices:
        lines.append(f"vertex {{ x = {float(vx)}; y = {float(vy)}; }}")
    lines.append(
        'sector { texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1"; '
        "heightfloor = 0; heightceiling = 128; lightlevel = 192; }"
    )
    lines.append(
        'sector { texturefloor = "GATE3"; textureceiling = "CEIL1_1"; '
        "heightfloor = 0; heightceiling = 128; lightlevel = 224; }"
    )
    for _ in range(2, next_sec):
        lines.append(
            'sector { texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1"; '
            "heightfloor = 0; heightceiling = 0; lightlevel = 192; }"
        )
    lines.extend(sidedefs)
    lines.extend(linedefs)
    px, py = player_pos
    lines.append(
        f"thing {{ x = {px}; y = {py}; angle = {player_angle}; type = 1; "
        "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }"
    )
    for kx, ky, ktype in key_things:
        lines.append(
            f"thing {{ x = {kx}; y = {ky}; angle = 0; type = {ktype}; "
            "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }"
        )
    for r, c, kind, ang in extra_things or []:
        tx = (c + 0.5) * cell_size
        ty = (h - r - 0.5) * cell_size
        lines.append(
            f"thing {{ x = {tx}; y = {ty}; angle = {ang}; type = {kind}; "
            "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }"
        )
    return "\n".join(lines) + "\n"


def _carve_l_corridor(grid: list[list[str]], a: tuple[int, int], b: tuple[int, int], rng: np.random.Generator) -> None:
    r1, c1 = a
    r2, c2 = b
    horiz_first = bool(rng.integers(0, 2))
    if horiz_first:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            grid[r1][c] = "."
        for r in range(min(r1, r2), max(r1, r2) + 1):
            grid[r][c2] = "."
    else:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            grid[r][c1] = "."
        for c in range(min(c1, c2), max(c1, c2) + 1):
            grid[r2][c] = "."


def _generate_candidate(
    task: str,
    tier: int,
    seed: int,
) -> tuple[list[str], int, list[tuple[int, int, int, int]], dict] | None:
    rng = np.random.default_rng(seed)
    h = int(rng.choice([9, 10, 11, 12]))
    w = int(rng.choice([10, 11, 12, 13, 14]))
    grid = [["#" for _ in range(w)] for _ in range(h)]

    n_rooms = 2 if tier == 0 else int(rng.choice([3, 4]))
    rooms: list[tuple[int, int, int, int]] = []  # (r0, c0, rh, cw)
    for _ in range(40):
        if len(rooms) >= n_rooms:
            break
        rh = int(rng.integers(2, 5))
        cw = int(rng.integers(2, 5))
        if h - rh - 1 < 1 or w - cw - 1 < 1:
            continue
        r0 = int(rng.integers(1, h - rh))
        c0 = int(rng.integers(1, w - cw))
        # Ensure at least 1 wall cell between distinct rooms so doors can sit on boundaries
        overlap = any(
            not (r0 + rh + 1 <= rr0 or rr0 + rrh + 1 <= r0 or c0 + cw + 1 <= cc0 or cc0 + ccw + 1 <= c0)
            for rr0, cc0, rrh, ccw in rooms
        )
        if not overlap:
            rooms.append((r0, c0, rh, cw))
            for r in range(r0, r0 + rh):
                for c in range(c0, c0 + cw):
                    grid[r][c] = "."
            # Optional pillar inside large rooms (>= 3x3) for navigation/combat variety
            if rh >= 3 and cw >= 3 and rng.random() < 0.45:
                grid[r0 + rh // 2][c0 + cw // 2] = "#"

    if len(rooms) < 2:
        return None

    def room_center(rm: tuple[int, int, int, int]) -> tuple[int, int]:
        r0, c0, rh, cw = rm
        pts = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + cw) if grid[r][c] == "."]
        return pts[int(rng.integers(0, len(pts)))]

    # Build spanning tree of rooms
    centers = [room_center(rm) for rm in rooms]
    for i in range(1, len(rooms)):
        parent = int(rng.integers(0, i))
        _carve_l_corridor(grid, centers[parent], centers[i], rng)

    # Optional loop or dead-end branch for navigation/combat/progression variety
    if task in ("navigation", "combat") and len(rooms) >= 3 and rng.random() < 0.6:
        a, b = rng.choice(len(rooms), size=2, replace=False)
        _carve_l_corridor(grid, centers[int(a)], centers[int(b)], rng)

    # Optional dead-end alcove
    if rng.random() < 0.6:
        base_r, base_c = centers[int(rng.integers(0, len(rooms)))]
        dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][int(rng.integers(0, 4))]
        length = int(rng.integers(2, 4))
        for k in range(1, length + 1):
            nr, nc = base_r + dr * k, base_c + dc * k
            if 1 <= nr < h - 1 and 1 <= nc < w - 1:
                grid[nr][nc] = "."

    # Pick Player P and Exit E
    p_r0, p_c0, p_rh, p_cw = rooms[0]
    p_cands = [(r, c) for r in range(p_r0, p_r0 + p_rh) for c in range(p_c0, p_c0 + p_cw) if grid[r][c] == "."]
    if not p_cands:
        return None
    pr, pc = p_cands[int(rng.integers(0, len(p_cands)))]
    grid[pr][pc] = "P"

    str_grid = ["".join(row) for row in grid]
    dist_all = bfs_distances(str_grid, (pr, pc))
    e_r0, e_c0, e_rh, e_cw = rooms[-1]
    e_cands = [
        (r, c)
        for r in range(e_r0, e_r0 + e_rh)
        for c in range(e_c0, e_c0 + e_cw)
        if grid[r][c] == "." and dist_all.get((r, c), 0) >= (4 if tier == 0 else 6)
    ]
    if not e_cands:
        return None
    er, ec = e_cands[int(rng.integers(0, len(e_cands)))]
    grid[er][ec] = "E"

    # Choose player_angle pointing roughly toward E or a random cardinal direction
    angles = [0, 90, 180, 270]
    player_angle = int(angles[int(rng.integers(0, 4))])

    extra_things: list[tuple[int, int, int, int]] = []
    meta: dict = {
        "grid_h": h,
        "grid_w": w,
        "num_rooms": len(rooms),
        "tier": tier,
        "route_len": dist_all[(er, ec)],
    }

    if task == "unlocked_door":
        # Find a properly framed bottleneck '.' cell that separates P from E
        str_grid = ["".join(row) for row in grid]
        bottlenecks = [
            (r, c)
            for (r, c), d in dist_all.items()
            if grid[r][c] == "." and d >= 2 and _door_properly_framed(str_grid, r, c)
        ]
        rng.shuffle(bottlenecks)
        placed = False
        for dr, dc in bottlenecks:
            d_blocked = bfs_distances(str_grid, (pr, pc), blocked_chars={"#", "D"})
            grid[dr][dc] = "D"
            test_grid = ["".join(row) for row in grid]
            d_check = bfs_distances(test_grid, (pr, pc), blocked_chars={"#", "D"})
            if (er, ec) not in d_check:
                placed = True
                break
            grid[dr][dc] = "."
        if not placed:
            return None

    elif task == "progression":
        key_ch, door_ch, color = [("r", "R", "red"), ("b", "B", "blue"), ("y", "Y", "yellow")][
            int(rng.integers(0, 3))
        ]
        str_grid = ["".join(row) for row in grid]
        bottlenecks = [
            (r, c)
            for (r, c), d in dist_all.items()
            if grid[r][c] == "." and d >= 2 and _door_properly_framed(str_grid, r, c)
        ]
        rng.shuffle(bottlenecks)
        placed = False
        for dr, dc in bottlenecks:
            grid[dr][dc] = door_ch
            test_grid = ["".join(row) for row in grid]
            before_door = bfs_distances(test_grid, (pr, pc), blocked_chars={"#", door_ch})
            if (er, ec) not in before_door:
                # Key must be reachable from P BEFORE crossing the locked door
                key_cands = [
                    (r, c)
                    for (r, c), d in before_door.items()
                    if grid[r][c] == "." and d >= 2
                ]
                if key_cands:
                    # Prefer farthest reachable cell in P's subgraph (often a branch/dead-end)
                    key_cands.sort(key=lambda rc: (-before_door[rc], rc))
                    kr, kc = key_cands[0]
                    grid[kr][kc] = key_ch
                    meta["key_color"] = color
                    meta["key_dist"] = before_door[(kr, kc)]
                    placed = True
                    break
            grid[dr][dc] = "."
        if not placed:
            return None

    elif task == "combat":
        # Half with an unlocked door, half open/looped
        use_door = bool(seed % 2 == 0)
        str_grid = ["".join(row) for row in grid]
        if use_door:
            bottlenecks = [
                (r, c)
                for (r, c), d in dist_all.items()
                if grid[r][c] == "." and d >= 2 and _door_properly_framed(str_grid, r, c)
            ]
            rng.shuffle(bottlenecks)
            for dr, dc in bottlenecks:
                grid[dr][dc] = "D"
                test_grid = ["".join(row) for row in grid]
                if (er, ec) not in bfs_distances(test_grid, (pr, pc), blocked_chars={"#", "D"}):
                    break
                grid[dr][dc] = "."
        str_grid = ["".join(row) for row in grid]
        floors = [
            (r, c)
            for (r, c), d in dist_all.items()
            if grid[r][c] == "." and d >= 3
        ]
        if len(floors) < 2:
            return None
        floors.sort(key=lambda rc: (-dist_all[rc], rc))
        n_enemies = 1 if tier == 0 else int(rng.choice([1, 2]))
        enemies = floors[:n_enemies]
        for j, (r, c) in enumerate(enemies):
            kind = 3004 if j == 0 else int(rng.choice([3004, 3001]))
            extra_things.append((r, c, kind, (player_angle + 180) % 360))
        spare = [rc for rc in dist_all if grid[rc[0]][rc[1]] == "." and rc not in enemies and dist_all[rc] >= 1]
        for (r, c), kind in zip(spare, [2007, 2011]):
            extra_things.append((r, c, kind, 0))
        meta["enemies"] = n_enemies
        meta["has_door"] = any("D" in row for row in grid)

    final_grid = ["".join(row) for row in grid]
    return final_grid, player_angle, extra_things, meta


def generate(
    train_per_task: int = 16,
    val_per_task: int = 4,
    test_per_task: int = 4,
) -> dict:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    seen_d4: set[tuple[str, ...]] = set()
    maps: list[dict] = []

    split_specs = [
        ("train", train_per_task, 1000),
        ("val", val_per_task, 10000),
        ("test", test_per_task, 20000),
    ]
    tasks = ["navigation", "unlocked_door", "progression", "combat"]

    for split, count_per_task, base_seed in split_specs:
        for t_idx, task in enumerate(tasks):
            accepted = 0
            attempt = 0
            while accepted < count_per_task:
                if attempt > 2000:
                    raise RuntimeError(f"Failed to generate {count_per_task} unique maps for {split}/{task}")
                seed = base_seed + t_idx * 1000 + attempt
                attempt += 1
                tier = 0 if accepted < count_per_task // 2 else 1
                cand = _generate_candidate(task, tier, seed)
                if cand is None:
                    continue
                grid, angle, things, meta = cand
                sig = d4_canonical(grid)
                if sig in seen_d4:
                    continue
                seen_d4.add(sig)

                name = f"proc_{split}_{task}_{accepted:02d}"
                text = procedural_grid_to_udmf(grid, player_angle=angle, extra_things=things)
                path = MAPS_DIR / f"{name}.wad"
                write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
                entry = {
                    "name": name,
                    "split": split,
                    "task": task,
                    "map_name": "MAP01",
                    "generator_version": GENERATOR_VERSION,
                    "seed": seed,
                    "params": meta,
                    "player_angle": angle,
                    "grid": grid,
                    "things": [{"r": r, "c": c, "type": k, "angle": a} for r, c, k, a in things],
                    "wad_path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sampling_weight": TASK_WEIGHTS[task] / count_per_task if split == "train" else 1.0,
                }
                if "key_color" in meta:
                    entry["key_color"] = meta["key_color"]
                if "enemies" in meta:
                    entry["enemies"] = meta["enemies"]
                if "has_door" in meta:
                    entry["has_door"] = meta["has_door"]
                maps.append(entry)
                accepted += 1

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "task_probabilities": TASK_WEIGHTS,
        "counts": {
            "train_per_task": train_per_task,
            "val_per_task": val_per_task,
            "test_per_task": test_per_task,
            "total": len(maps),
        },
        "maps": maps,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    m = generate()
    print(f"Generated {len(m['maps'])} procedural maps ({m['generator_version']})")
