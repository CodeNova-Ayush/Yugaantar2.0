"""Generate a varied 20-map navigation curriculum (12 train, 4 val, 4 test).

Splits are separated by underlying geometry before rotation or mirroring so no
transformed duplicate crosses splits. All maps use vanilla-compatible walk-over
exits (UDMF special=243 Exit_Normal, playercross=true) with no enemies, keys,
or doors. Legacy maps in training/maps/ remain untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env.wad import Lump, write_wad  # noqa: E402
from training.generate_maps import CELL_SIZE, grid_to_udmf, sha256_file  # noqa: E402

CURRICULUM_DIR = Path(__file__).resolve().parent
MAPS_DIR = CURRICULUM_DIR / "maps"

# Angles in Doom: 0=East, 90=North, 180=West, 270=South.
CURRICULUM_SPECS = [
    # ------------------------------------------------------------------ TRAIN (12)
    {
        "name": "train_straight_short_e",
        "split": "train",
        "player_angle": 0,
        "geometry_family": "straight_5x2",
        "grid": [
            "#######",
            "#P...E#",
            "#....E#",
            "#######",
        ],
    },
    {
        "name": "train_straight_long_w",
        "split": "train",
        "player_angle": 180,
        "geometry_family": "straight_8x2",
        "grid": [
            "##########",
            "#E......P#",
            "#E.......#",
            "##########",
        ],
    },
    {
        "name": "train_wide_hall_n",
        "split": "train",
        "player_angle": 90,
        "geometry_family": "wide_hall_4x5",
        "grid": [
            "######",
            "#EEEE#",
            "#....#",
            "#....#",
            "#....#",
            "#..P.#",
            "######",
        ],
    },
    {
        "name": "train_bend_left_90_e",
        "split": "train",
        "player_angle": 0,
        "geometry_family": "l_bend_5x4",
        "grid": [
            "#######",
            "###EEE#",
            "###...#",
            "#P....#",
            "#.....#",
            "#######",
        ],
    },
    {
        "name": "train_bend_right_90_s",
        "split": "train",
        "player_angle": 270,
        "geometry_family": "l_bend_5x4",
        "grid": [
            "#######",
            "###.P.#",
            "###...#",
            "#E....#",
            "#E....#",
            "#######",
        ],
    },
    {
        "name": "train_bend_wide_left_n",
        "split": "train",
        "player_angle": 90,
        "geometry_family": "l_bend_wide_3cell",
        "grid": [
            "########",
            "#E.....#",
            "#E.....#",
            "#E.....#",
            "####...#",
            "####.P.#",
            "########",
        ],
    },
    {
        "name": "train_s_curve_e",
        "split": "train",
        "player_angle": 0,
        "geometry_family": "s_curve_compact",
        "grid": [
            "#########",
            "###....E#",
            "###....E#",
            "#P...####",
            "#....####",
            "#########",
        ],
    },
    {
        "name": "train_s_curve_long_s",
        "split": "train",
        "player_angle": 270,
        "geometry_family": "s_curve_long",
        "grid": [
            "#######",
            "#P.####",
            "#..####",
            "#.....#",
            "#.....#",
            "####..#",
            "####EE#",
            "#######",
        ],
    },
    {
        "name": "train_u_turn_e",
        "split": "train",
        "player_angle": 0,
        "geometry_family": "u_turn_2cell",
        "grid": [
            "#######",
            "#E....#",
            "#E....#",
            "###...#",
            "#P....#",
            "#.....#",
            "#######",
        ],
    },
    {
        "name": "train_t_junction_n",
        "split": "train",
        "player_angle": 90,
        "geometry_family": "t_junction_deadend",
        "grid": [
            "#########",
            "#..#....#",
            "#..#...E#",
            "#......E#",
            "###..####",
            "###P.####",
            "#########",
        ],
    },
    {
        "name": "train_staggered_rooms_e",
        "split": "train",
        "player_angle": 0,
        "geometry_family": "two_rooms_chokepoint",
        "grid": [
            "#########",
            "#...#..E#",
            "#P......#",
            "#...#...#",
            "#########",
        ],
    },
    {
        "name": "train_stepped_stair_n",
        "split": "train",
        "player_angle": 90,
        "geometry_family": "stepped_stair_3turn",
        "grid": [
            "########",
            "####..E#",
            "####..E#",
            "##....##",
            "##....##",
            "#..#####",
            "#P.#####",
            "########",
        ],
    },
    # ------------------------------------------------------------------ VAL (4)
    {
        "name": "val_narrow_long_n",
        "split": "val",
        "player_angle": 90,
        "geometry_family": "narrow_1cell_7",
        "grid": [
            "###",
            "#E#",
            "#.#",
            "#.#",
            "#.#",
            "#.#",
            "#.#",
            "#P#",
            "###",
        ],
    },
    {
        "name": "val_l_asymmetric_w",
        "split": "val",
        "player_angle": 180,
        "geometry_family": "l_asymmetric_3x6",
        "grid": [
            "#####",
            "#..P#",
            "#...#",
            "#..##",
            "#..##",
            "#..##",
            "#EE##",
            "#####",
        ],
    },
    {
        "name": "val_pillar_bypass_e",
        "split": "val",
        "player_angle": 0,
        "geometry_family": "pillar_bypass_loop",
        "grid": [
            "########",
            "#......#",
            "#P.##.E#",
            "#..##.E#",
            "#......#",
            "########",
        ],
    },
    {
        "name": "val_hook_chamber_s",
        "split": "val",
        "player_angle": 270,
        "geometry_family": "hook_chamber_3x2",
        "grid": [
            "#######",
            "####.P#",
            "####..#",
            "#..#..#",
            "#E....#",
            "#.....#",
            "#######",
        ],
    },
    # ------------------------------------------------------------------ TEST (4)
    {
        "name": "test_chicane_e",
        "split": "test",
        "player_angle": 0,
        "geometry_family": "double_chicane",
        "grid": [
            "##########",
            "###....###",
            "###.##.###",
            "#P..##..E#",
            "#...##..E#",
            "##########",
        ],
    },
    {
        "name": "test_l_narrow_to_wide_s",
        "split": "test",
        "player_angle": 270,
        "geometry_family": "l_narrow_to_wide",
        "grid": [
            "#######",
            "#P#####",
            "#.#####",
            "#.#####",
            "#....E#",
            "#....E#",
            "#....E#",
            "#######",
        ],
    },
    {
        "name": "test_fork_deadend_w",
        "split": "test",
        "player_angle": 180,
        "geometry_family": "fork_deadend_y",
        "grid": [
            "########",
            "#..#####",
            "#......#",
            "####..P#",
            "#E.....#",
            "#E.#####",
            "########",
        ],
    },
    {
        "name": "test_spiral_inward_n",
        "split": "test",
        "player_angle": 90,
        "geometry_family": "spiral_inward",
        "grid": [
            "#######",
            "#.....#",
            "#.##..#",
            "#.#E..#",
            "#.#.#.#",
            "#P..#.#",
            "#######",
        ],
    },
]


def build_curriculum() -> dict:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cell_size": CELL_SIZE,
        "exit_mechanism": "UDMF linedef special=243 (Exit_Normal), playercross=true",
        "note": "Rays do not explicitly identify exits; exits are walk-over linedefs into GATE3.",
        "maps": [],
    }
    for spec in CURRICULUM_SPECS:
        grid_str = "\n".join(spec["grid"])
        wad_path = MAPS_DIR / f"{spec['name']}.wad"
        textmap = grid_to_udmf(grid_str, player_angle=spec["player_angle"])
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
                "geometry_family": spec["geometry_family"],
                "sha256": sha256_file(wad_path),
                "grid": spec["grid"],
                "player_angle": spec["player_angle"],
            }
        )

    (CURRICULUM_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    info = build_curriculum()
    counts = {}
    for m in info["maps"]:
        counts[m["split"]] = counts.get(m["split"], 0) + 1
    print(f"Generated {len(info['maps'])} curriculum maps: {counts}")
