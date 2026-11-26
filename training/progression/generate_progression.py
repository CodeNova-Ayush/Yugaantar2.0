"""Generate the Stage 5A key-and-door progression benchmark (6 train, 2 val, 2 test).

Uses vanilla Doom keycards and UDMF Door_LockedRaise (special=13) with
walk-over Exit_Normal (special=243). Splits are separated by D4-canonicalized
walkable geometry (ignoring P, E, keys, and doors).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env.wad import Lump, write_wad  # noqa: E402
from training.generate_maps import CELL_SIZE, sha256_file  # noqa: E402

PROG_DIR = Path(__file__).resolve().parent
MAPS_DIR = PROG_DIR / "maps"

KEY_SPEC = {
    "r": {"thing_type": 13, "vital_name": "key_red_card", "color": "red"},
    "b": {"thing_type": 5, "vital_name": "key_blue_card", "color": "blue"},
    "y": {"thing_type": 6, "vital_name": "key_yellow_card", "color": "yellow"},
}

DOOR_SPEC = {
    "R": {"lock_arg": 129, "texture": "DOORRED", "key_char": "r", "color": "red"},
    "B": {"lock_arg": 130, "texture": "DOORBLU", "key_char": "b", "color": "blue"},
    "Y": {"lock_arg": 131, "texture": "DOORYEL", "key_char": "y", "color": "yellow"},
}

PROGRESSION_SPECS = [
    # ------------------------------------------------------------------ TRAIN (6)
    {
        "name": "prog_train_inline_red_e",
        "split": "train",
        "player_angle": 0,
        "key_color": "red",
        "door_occluded_at_key": False,
        "grid": [
            "###########",
            "#P.....R.E#",
            "#..r...####",
            "###########",
        ],
    },
    {
        "name": "prog_train_side_north_e",
        "split": "train",
        "player_angle": 0,
        "key_color": "red",
        "out_of_door_range": True,
        "grid": [
            "#########",
            "#..r#####",
            "#..######",
            "#P..R..E#",
            "#...#####",
            "#########",
        ],
    },
    {
        "name": "prog_train_side_south_w",
        "split": "train",
        "player_angle": 180,
        "key_color": "blue",
        "out_of_door_range": True,
        "grid": [
            "##########",
            "#E..B...P#",
            "#####....#",
            "#####..###",
            "#####b.###",
            "##########",
        ],
    },
    {
        "name": "prog_train_hook_yellow_n",
        "split": "train",
        "player_angle": 90,
        "key_color": "yellow",
        "door_occluded_at_key": False,
        "grid": [
            "########",
            "#E######",
            "#Y######",
            "#...####",
            "#P.....#",
            "####..y#",
            "########",
        ],
    },
    {
        "name": "prog_train_wide_hall_red_s",
        "split": "train",
        "player_angle": 270,
        "key_color": "red",
        "door_occluded_at_key": False,
        "grid": [
            "#########",
            "#r.######",
            "#..######",
            "#....P..#",
            "#.......#",
            "####R####",
            "####E####",
            "#########",
        ],
    },
    {
        "name": "prog_train_deep_branch_blue_e",
        "split": "train",
        "player_angle": 0,
        "key_color": "blue",
        "out_of_door_range": True,
        "grid": [
            "############",
            "#b.....#####",
            "######.#####",
            "######.#####",
            "#P......B.E#",
            "#.......####",
            "############",
        ],
    },
    # ------------------------------------------------------------------ VAL (2)
    {
        "name": "prog_val_t_branch_e",
        "split": "val",
        "player_angle": 0,
        "key_color": "blue",
        "out_of_door_range": True,
        "grid": [
            "############",
            "###b..######",
            "#####.######",
            "#####.######",
            "#P......B.E#",
            "############",
        ],
    },
    {
        "name": "prog_val_u_detour_s",
        "split": "val",
        "player_angle": 270,
        "key_color": "yellow",
        "out_of_door_range": True,
        "grid": [
            "#########",
            "#####..y#",
            "#P..#...#",
            "#.......#",
            "#..######",
            "#Y#######",
            "#E#######",
            "#########",
        ],
    },
    # ------------------------------------------------------------------ TEST (2)
    {
        "name": "prog_test_cross_branch_w",
        "split": "test",
        "player_angle": 180,
        "key_color": "red",
        "out_of_door_range": True,
        "grid": [
            "###########",
            "#E.R....P.#",
            "######...##",
            "######.####",
            "######...r#",
            "###########",
        ],
    },
    {
        "name": "prog_test_long_loop_n",
        "split": "test",
        "player_angle": 90,
        "key_color": "blue",
        "out_of_door_range": True,
        "grid": [
            "##########",
            "#####E####",
            "#####B####",
            "#....P...#",
            "#.########",
            "#.########",
            "#.....b###",
            "##########",
        ],
    },
]


def progression_grid_to_udmf(grid: list[str], cell_size: int = CELL_SIZE, player_angle: int = 0) -> str:
    """Convert an ASCII progression grid into UDMF TEXTMAP with locked door and keycard."""
    h = len(grid)
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        raise ValueError("All grid rows must have equal width")

    # Sector 0 = main floor, Sector 1 = exit (GATE3), Sectors 2+ = locked door sectors.
    door_sectors: dict[tuple[int, int], tuple[int, str]] = {}
    next_sec = 2
    for r in range(h):
        for c in range(w):
            ch = grid[r][c]
            if ch in DOOR_SPEC:
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

    def add_locked_door_line(x1: int, y1: int, x2: int, y2: int, front_sec: int, door_sec: int, door_ch: str) -> None:
        spec = DOOR_SPEC[door_ch]
        sf = add_sidedef(front_sec, top=spec["texture"])
        sb = add_sidedef(door_sec)
        v1, v2 = vid(x1, y1), vid(x2, y2)
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
                    tex = "DOORTRAK" if sec >= 2 else ("GRAY1" if sec == 1 else "STARTAN2")
                    add_wall(ax, ay, bx, by, sec, tex=tex)
                elif sec != 1 and nsec == 1:
                    add_exit_line(ax, ay, bx, by, sec)
                elif sec < 2 and nsec >= 2:
                    _, dch = door_sectors[(nr, nc)]
                    add_locked_door_line(ax, ay, bx, by, sec, nsec, dch)

    assert player_pos is not None, "Grid must contain 'P'"
    assert len(key_things) == 1, "Grid must contain exactly 1 keycard"
    assert len(door_sectors) == 1, "Grid must contain exactly 1 locked door"

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
            'sector { texturefloor = "FLAT5_4"; textureceiling = "CEIL1_1"; '
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
    return "\n".join(lines) + "\n"


def build_progression() -> dict:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cell_size": CELL_SIZE,
        "door_mechanism": "UDMF Door_LockedRaise (special=13, arg3=129/130/131 for R/B/Y, playeruse=true)",
        "exit_mechanism": "UDMF linedef special=243 (Exit_Normal), playercross=true",
        "maps": [],
    }
    for spec in PROGRESSION_SPECS:
        wad_path = MAPS_DIR / f"{spec['name']}.wad"
        textmap = progression_grid_to_udmf(spec["grid"], player_angle=spec["player_angle"])
        write_wad(
            wad_path,
            [
                Lump("MAP01", b""),
                Lump("TEXTMAP", textmap.encode("utf-8")),
                Lump("ENDMAP", b""),
            ],
        )
        manifest["maps"].append(
            {
                "name": spec["name"],
                "wad_path": str(wad_path.relative_to(ROOT)),
                "map_name": "MAP01",
                "split": spec["split"],
                "key_color": spec["key_color"],
                "door_occluded_at_key": spec.get("door_occluded_at_key", True),
                "sha256": sha256_file(wad_path),
                "grid": spec["grid"],
                "player_angle": spec["player_angle"],
            }
        )

    (PROG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    info = build_progression()
    counts = {}
    for m in info["maps"]:
        counts[m["split"]] = counts.get(m["split"], 0) + 1
    print(f"Generated {len(info['maps'])} progression maps: {counts}")
