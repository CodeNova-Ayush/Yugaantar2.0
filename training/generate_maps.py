"""Generate reproducible UDMF practice and held-out navigation maps.

Maps use vanilla-compatible walk-over exits (ZDoom UDMF linedef special 243:
Exit_Normal with playercross = true). Crossing the boundary into the exit
sector (floor texture GATE3) immediately ends the episode with exit_reached=True.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from doom_env.wad import Lump, write_wad

ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = Path(__file__).resolve().parent / "maps"

CELL_SIZE = 64

# Practice layout: straight 2-cell-wide corridor (easy navigation, no turn needed).
PRACTICE_GRID = "\n".join(
    [
        "#######",
        "#P...E#",
        "#....E#",
        "#######",
    ]
)

# Held-out layout: L-shaped 2-cell-wide corridor (requires a 90-degree left turn).
HELDOUT_GRID = "\n".join(
    [
        "#######",
        "###EEE#",
        "###...#",
        "#P....#",
        "#.....#",
        "#######",
    ]
)


def grid_to_udmf(grid: str, cell_size: int = CELL_SIZE, player_angle: int = 0) -> str:
    """Convert an ASCII grid (# wall, . floor, P player, E exit) into UDMF TEXTMAP."""
    rows = [line.strip() for line in grid.strip().splitlines() if line.strip()]
    h = len(rows)
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("All grid rows must have equal width")

    def cell_sector(r: int, c: int) -> int | None:
        if r < 0 or r >= h or c < 0 or c >= w:
            return None
        ch = rows[r][c]
        if ch in (".", "P"):
            return 0
        if ch == "E":
            return 1
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

    def add_sidedef(sector: int, middle: str = "-") -> int:
        idx = len(sidedefs)
        sidedefs.append(
            f'sidedef {{ sector = {sector}; texturetop = "-"; '
            f'texturemiddle = "{middle}"; texturebottom = "-"; }}'
        )
        return idx

    def add_wall(x1: int, y1: int, x2: int, y2: int, sector: int) -> None:
        tex = "STONE2" if sector == 1 else "STARTAN2"
        s = add_sidedef(sector, middle=tex)
        v1, v2 = vid(x1, y1), vid(x2, y2)
        linedefs.append(
            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {s}; "
            f"blocking = true; dontpegbottom = true; }}"
        )

    def add_exit_line(x1: int, y1: int, x2: int, y2: int) -> None:
        sf = add_sidedef(0, middle="-")
        sb = add_sidedef(1, middle="-")
        v1, v2 = vid(x1, y1), vid(x2, y2)
        linedefs.append(
            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; "
            f"twosided = true; special = 243; playercross = true; }}"
        )

    player_pos: tuple[float, float] | None = None
    for r in range(h):
        for c in range(w):
            sec = cell_sector(r, c)
            if sec is None:
                continue
            x0, x1 = c * cell_size, (c + 1) * cell_size
            # Flip row so row 0 is at the top (North, +Y).
            y0, y1 = (h - 1 - r) * cell_size, (h - r) * cell_size
            if rows[r][c] == "P":
                player_pos = (x0 + cell_size / 2.0, y0 + cell_size / 2.0)

            # Check each of the 4 cardinal neighbors with clockwise front-side orientation:
            # North (r-1): (x0, y1) -> (x1, y1)
            # East  (c+1): (x1, y1) -> (x1, y0)
            # South (r+1): (x1, y0) -> (x0, y0)
            # West  (c-1): (x0, y0) -> (x0, y1)
            neighbors = [
                (r - 1, c, x0, y1, x1, y1),
                (r, c + 1, x1, y1, x1, y0),
                (r + 1, c, x1, y0, x0, y0),
                (r, c - 1, x0, y0, x0, y1),
            ]
            for nr, nc, ax, ay, bx, by in neighbors:
                nsec = cell_sector(nr, nc)
                if nsec is None:
                    add_wall(ax, ay, bx, by, sec)
                elif sec == 0 and nsec == 1:
                    add_exit_line(ax, ay, bx, by)

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
    lines.extend(sidedefs)
    lines.extend(linedefs)
    px, py = player_pos
    lines.append(
        f"thing {{ x = {px}; y = {py}; angle = {player_angle}; type = 1; "
        "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }"
    )
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_maps() -> dict:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "name": "practice_corridor",
            "filename": "practice_corridor.wad",
            "map_name": "MAP01",
            "split": "practice",
            "grid": PRACTICE_GRID,
            "player_angle": 0,
            "description": "Straight 2-cell-wide corridor from West to East; no enemies or doors; walk-over Exit_Normal (special 243, playercross=true).",
        },
        {
            "name": "heldout_lshape",
            "filename": "heldout_lshape.wad",
            "map_name": "MAP01",
            "split": "heldout",
            "grid": HELDOUT_GRID,
            "player_angle": 0,
            "description": "L-shaped 2-cell-wide corridor (East then North 90-degree turn); no enemies or doors; walk-over Exit_Normal (special 243, playercross=true).",
        },
    ]

    manifest = {"cell_size": CELL_SIZE, "exit_mechanism": "UDMF linedef special=243 (Exit_Normal), playercross=true", "maps": []}
    for spec in specs:
        wad_path = MAPS_DIR / spec["filename"]
        textmap = grid_to_udmf(spec["grid"], player_angle=spec["player_angle"])
        write_wad(
            wad_path,
            [
                Lump(spec["map_name"], b""),
                Lump("TEXTMAP", textmap.encode("utf-8")),
                Lump("ENDMAP", b""),
            ],
        )
        manifest["maps"].append(
            {
                "name": spec["name"],
                "wad_path": str(wad_path.relative_to(ROOT)),
                "map_name": spec["map_name"],
                "split": spec["split"],
                "sha256": sha256_file(wad_path),
                "grid": spec["grid"].splitlines(),
                "player_angle": spec["player_angle"],
                "description": spec["description"],
            }
        )

    manifest_path = MAPS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    info = build_maps()
    for m in info["maps"]:
        print(f"{m['name']} ({m['split']}): {m['wad_path']} sha256={m['sha256']}")
