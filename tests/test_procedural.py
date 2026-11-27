"""Focused tests for the reproducible procedural map pipeline and verification checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from doom_env import make_env
from training.procedural.generate import ROOT, _generate_candidate, bfs_distances, d4_canonical
from training.progression.generate_progression import DOOR_SPEC


def test_procedural_generator_determinism():
    c1 = _generate_candidate("progression", tier=1, seed=12345)
    c2 = _generate_candidate("progression", tier=1, seed=12345)
    assert c1 is not None and c2 is not None
    assert c1 == c2


def test_procedural_manifest_splits_and_solvability():
    manifest = json.loads((ROOT / "training/procedural/manifest.json").read_text())
    maps = manifest["maps"]
    assert len(maps) == 96

    sigs = set()
    for m in maps:
        grid = m["grid"]
        sig = d4_canonical(grid)
        assert sig not in sigs, f"Duplicate D4 signature for {m['name']}"
        sigs.add(sig)

        pr, pc = next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "P")
        er, ec = next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "E")
        assert (er, ec) in bfs_distances(grid, (pr, pc))

        if m["task"] == "progression":
            dch = next(ch for row in grid for ch in row if ch in DOOR_SPEC)
            kch = DOOR_SPEC[dch]["key_char"]
            kpos = next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == kch)
            before = bfs_distances(grid, (pr, pc), blocked_chars={"#", dch})
            assert (er, ec) not in before
            assert kpos in before


def test_representative_procedural_maps_engine_load_and_exit():
    manifest = json.loads((ROOT / "training/procedural/manifest.json").read_text())
    # Pick one navigation and one unlocked_door map and verify WAD loads and action stepping works
    for task in ("navigation", "unlocked_door", "progression", "combat"):
        m = next(x for x in manifest["maps"] if x["task"] == task and x["split"] == "train")
        env = make_env(ROOT / m["wad_path"], m["map_name"], episode_ticks=2100, render_buffers=False)
        try:
            obs, _ = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            act = np.zeros(9, dtype=np.int8)
            act[0] = 1
            obs2, _, _, _, _ = env.step(act)
            assert env.observation_space.contains(obs2)
        finally:
            env.close()
