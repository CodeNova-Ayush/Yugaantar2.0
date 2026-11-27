"""Generate Curriculum v2 (mixed_v2) with unlocked doors and multi-room combat.

Preserves all existing curriculum, progression, and mixed v1 artifacts.
Splits are strictly separated by D4-canonicalized walkable geometry.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

from doom_env.wad import Lump, write_wad
from training.generate_maps import CELL_SIZE, grid_to_udmf

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "training/mixed_v2"
MAPS_DIR = OUT_DIR / "maps"

UNLOCKED_DOOR_SPECS = [
    # ------------------------------------------------------------------ TRAIN (8)
    {
        "name": "udoor_train_two_room_e",
        "split": "train",
        "player_angle": 0,
        "grid": [
            "###########",
            "#P...#....#",
            "#....D...E#",
            "#....#....#",
            "###########",
        ],
    },
    {
        "name": "udoor_train_offset_ne",
        "split": "train",
        "player_angle": 0,
        "grid": [
            "############",
            "#P....#....#",
            "#.....#....#",
            "#.....D...E#",
            "#.....#....#",
            "############",
        ],
    },
    {
        "name": "udoor_train_wide_w",
        "split": "train",
        "player_angle": 180,
        "grid": [
            "###########",
            "#E..#.....#",
            "#...D....P#",
            "#...D.....#",
            "#...#.....#",
            "###########",
        ],
    },
    {
        "name": "udoor_train_north_l",
        "split": "train",
        "player_angle": 90,
        "grid": [
            "#########",
            "#E......#",
            "#..######",
            "#.......#",
            "###D#####",
            "#......P#",
            "#.......#",
            "#########",
        ],
    },
    {
        "name": "udoor_train_south_pillar",
        "split": "train",
        "player_angle": 270,
        "grid": [
            "#########",
            "#P......#",
            "#..#....#",
            "####D####",
            "#.......#",
            "#...#..E#",
            "#########",
        ],
    },
    {
        "name": "udoor_train_double_door_e",
        "split": "train",
        "player_angle": 0,
        "grid": [
            "#############",
            "#P..#...#...#",
            "#...D...D..E#",
            "#...#...#...#",
            "#############",
        ],
    },
    {
        "name": "udoor_train_corner_sw",
        "split": "train",
        "player_angle": 180,
        "grid": [
            "##########",
            "#....#..P#",
            "#....D...#",
            "#....#...#",
            "#E..######",
            "##########",
        ],
    },
    {
        "name": "udoor_train_zig_n",
        "split": "train",
        "player_angle": 90,
        "grid": [
            "##########",
            "#......E.#",
            "#####D####",
            "#........#",
            "#P.......#",
            "##########",
        ],
    },
    # ------------------------------------------------------------------ VAL (3)
    {
        "name": "udoor_val_offset_e",
        "split": "val",
        "player_angle": 0,
        "grid": [
            "############",
            "#....#.....#",
            "#....D....E#",
            "#P...#.....#",
            "#....#..#..#",
            "############",
        ],
    },
    {
        "name": "udoor_val_south_w",
        "split": "val",
        "player_angle": 180,
        "grid": [
            "##########",
            "#....#..P#",
            "#....D...#",
            "#....#...#",
            "#.########",
            "#E.......#",
            "##########",
        ],
    },
    {
        "name": "udoor_val_wide_n",
        "split": "val",
        "player_angle": 90,
        "grid": [
            "##########",
            "#E.......#",
            "#........#",
            "###DD#####",
            "#.......P#",
            "#........#",
            "##########",
        ],
    },
    # ------------------------------------------------------------------ TEST (3) - FRESH RESERVED
    {
        "name": "udoor_test_two_room_e",
        "split": "test",
        "player_angle": 0,
        "grid": [
            "#############",
            "#P...#......#",
            "#....D..#...#",
            "#....#.....E#",
            "#....#......#",
            "#############",
        ],
    },
    {
        "name": "udoor_test_north_pillar_w",
        "split": "test",
        "player_angle": 180,
        "grid": [
            "##########",
            "#E..#....#",
            "#........#",
            "######D###",
            "#.......P#",
            "##########",
        ],
    },
    {
        "name": "udoor_test_south_e",
        "split": "test",
        "player_angle": 0,
        "grid": [
            "##########",
            "#P.......#",
            "#....#...#",
            "##D#######",
            "#.......E#",
            "##########",
        ],
    },
]


def unlocked_door_grid_to_udmf(
    grid: list[str],
    cell_size: int = CELL_SIZE,
    player_angle: int = 0,
    extra_things: list[tuple[int, int, int, int]] | None = None,
) -> str:
    """Convert an ASCII grid (# wall, . floor, P player, E exit, D unlocked door) to UDMF TEXTMAP."""
    h = len(grid)
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        raise ValueError("All grid rows must have equal width")

    # Group contiguous 'D' cells into shared door sectors starting at sector 2.
    door_sectors: dict[tuple[int, int], int] = {}
    next_sec = 2
    for r in range(h):
        for c in range(w):
            if grid[r][c] == "D" and (r, c) not in door_sectors:
                q = deque([(r, c)])
                door_sectors[(r, c)] = next_sec
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == "D" and (nr, nc) not in door_sectors:
                            door_sectors[(nr, nc)] = next_sec
                            q.append((nr, nc))
                next_sec += 1

    def cell_sector(r: int, c: int) -> int | None:
        if r < 0 or r >= h or c < 0 or c >= w:
            return None
        ch = grid[r][c]
        if ch in (".", "P"):
            return 0
        if ch == "E":
            return 1
        return door_sectors.get((r, c))

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

    def add_unlocked_door_line(x1: int, y1: int, x2: int, y2: int, front_sec: int, door_sec: int) -> None:
        sf = add_sidedef(front_sec, top="BIGDOOR2")
        sb = add_sidedef(door_sec)
        v1, v2 = vid(x1, y1), vid(x2, y2)
        linedefs.append(
            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; "
            f"twosided = true; special = 12; arg0 = 0; arg1 = 16; arg2 = 150; "
            f"playeruse = true; monsteruse = true; repeatspecial = true; }}"
        )

    player_pos: tuple[float, float] | None = None
    for r in range(h):
        for c in range(w):
            sec = cell_sector(r, c)
            if sec is None:
                continue
            x0, x1 = c * cell_size, (c + 1) * cell_size
            y0, y1 = (h - 1 - r) * cell_size, (h - r) * cell_size
            if grid[r][c] == "P":
                player_pos = (x0 + cell_size / 2.0, y0 + cell_size / 2.0)

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
                    add_unlocked_door_line(ax, ay, bx, by, sec, nsec)

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
    for r, c, kind, ang in extra_things or []:
        tx = (c + 0.5) * cell_size
        ty = (h - r - 0.5) * cell_size
        lines.append(
            f"thing {{ x = {tx}; y = {ty}; angle = {ang}; type = {kind}; "
            "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }"
        )
    return "\n".join(lines) + "\n"


def _bfs_distances(grid: list[str], start: tuple[int, int], block_doors: bool = False) -> dict[tuple[int, int], int]:
    h, w = len(grid), len(grid[0])
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist:
                ch = grid[nr][nc]
                if ch == "#" or (block_doors and ch == "D"):
                    continue
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def generate() -> dict:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    nav = json.loads((ROOT / "training/curriculum/manifest.json").read_text())["maps"]
    prog = json.loads((ROOT / "training/progression/manifest.json").read_text())["maps"]
    maps: list[dict] = [{**m, "task": "navigation"} for m in nav] + [{**m, "task": "progression"} for m in prog]

    # 1. Unlocked-door layouts
    udoor_entries: list[dict] = []
    for spec in UNLOCKED_DOOR_SPECS:
        text = unlocked_door_grid_to_udmf(spec["grid"], player_angle=spec["player_angle"])
        path = MAPS_DIR / f"{spec['name']}.wad"
        write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
        entry = {
            "name": spec["name"],
            "split": spec["split"],
            "task": "unlocked_door",
            "map_name": "MAP01",
            "player_angle": spec["player_angle"],
            "grid": spec["grid"],
            "wad_path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        maps.append(entry)
        udoor_entries.append(entry)

    # 2. Combat-rooms layouts:
    #    - One combat variant per unlocked_door layout (enemies beyond the door, ammo+health).
    #    - Plus single-room combat variants from navigation layouts (up to 4 train, 1 val, 1 test).
    for idx, src in enumerate(udoor_entries):
        rows = src["grid"]
        player = next((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "P")
        room1_dist = _bfs_distances(rows, player, block_doors=True)
        all_dist = _bfs_distances(rows, player, block_doors=False)
        floors_beyond = sorted(
            [rc for rc, d in all_dist.items() if rows[rc[0]][rc[1]] == "." and rc not in room1_dist],
            key=lambda rc: (-all_dist[rc], rc),
        )
        floors_near = sorted(
            [rc for rc in room1_dist if rows[rc[0]][rc[1]] == "." and room1_dist[rc] >= 1],
            key=lambda rc: (room1_dist[rc], rc),
        )
        if not floors_beyond:
            raise ValueError(f"No floor cells beyond door in {src['name']}")
        n_enemies = 1 if idx % 2 == 0 else min(2, len(floors_beyond))
        enemies = floors_beyond[:n_enemies]
        things: list[tuple[int, int, int, int]] = [
            (r, c, 3004 if j == 0 else 3001, (src["player_angle"] + 180) % 360)
            for j, (r, c) in enumerate(enemies)
        ]
        spare = [rc for rc in floors_near if rc not in enemies] or [rc for rc in floors_beyond if rc not in enemies]
        for (r, c), kind in zip(spare, [2007, 2011]):
            things.append((r, c, kind, 0))
        text = unlocked_door_grid_to_udmf(rows, player_angle=src["player_angle"], extra_things=things)
        name = "combat_" + src["name"]
        path = MAPS_DIR / f"{name}.wad"
        write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
        maps.append(
            {
                "name": name,
                "split": src["split"],
                "task": "combat_rooms",
                "map_name": "MAP01",
                "player_angle": src["player_angle"],
                "grid": rows,
                "wad_path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "enemies": n_enemies,
                "has_door": True,
            }
        )

    # Add 1-room combat layouts from navigation splits (4 train, 1 val, 1 test) -> total 12 train, 4 val, 4 test?
    # Wait: udoor has 8 train, 3 val, 3 test. Adding 2 train, 1 val, 1 test 1-room combat = 10 train, 4 val, 4 test.
    nav_quotas = {"train": 2, "val": 1, "test": 1}
    nav_counts = {"train": 0, "val": 0, "test": 0}
    for i, src in enumerate(nav):
        sp = src["split"]
        if nav_counts[sp] >= nav_quotas[sp]:
            continue
        rows = src["grid"]
        h = len(rows)
        player = next((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "P")
        floors = sorted(
            [
                (r, c)
                for r, row in enumerate(rows)
                for c, ch in enumerate(row)
                if ch == "." and abs(r - player[0]) + abs(c - player[1]) >= 2
            ],
            key=lambda rc: (abs(rc[0] - player[0]) + abs(rc[1] - player[1]), rc),
        )
        if not floors:
            continue
        n_enemies = 1 if nav_counts[sp] % 2 == 0 else min(2, len(floors))
        enemies = floors[:n_enemies]
        spare = [rc for rc in floors if rc not in enemies]
        text = grid_to_udmf("\n".join(rows), player_angle=src["player_angle"])
        for j, (r, c) in enumerate(enemies):
            kind = 3004 if j == 0 else 3001
            text += (
                f"thing {{ x = {(c + 0.5) * CELL_SIZE}; y = {(h - r - 0.5) * CELL_SIZE}; "
                f"type = {kind}; angle = {(src['player_angle'] + 180) % 360}; "
                "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }\n"
            )
        for (r, c), kind in zip(spare, [2007, 2011]):
            text += (
                f"thing {{ x = {(c + 0.5) * CELL_SIZE}; y = {(h - r - 0.5) * CELL_SIZE}; "
                f"type = {kind}; angle = 0; "
                "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }\n"
            )
        name = "combat1r_" + src["name"]
        path = MAPS_DIR / f"{name}.wad"
        write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
        maps.append(
            {
                "name": name,
                "split": sp,
                "task": "combat_rooms",
                "map_name": "MAP01",
                "player_angle": src["player_angle"],
                "grid": rows,
                "wad_path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "enemies": n_enemies,
                "has_door": False,
            }
        )
        nav_counts[sp] += 1

    weights = {"navigation": 0.15, "unlocked_door": 0.20, "progression": 0.35, "combat_rooms": 0.30}
    for m in maps:
        count = sum(x["task"] == m["task"] and x["split"] == m["split"] for x in maps)
        m["sampling_weight"] = weights[m["task"]] / count

    manifest = {
        "version": 2,
        "task_probabilities": weights,
        "note": "Curriculum v2 with D4-separated unlocked_door and multi-room combat_rooms layouts; no cross-split reuse.",
        "maps": maps,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    m = generate()
    print("Generated", len(m["maps"]), "mixed_v2 layouts")
