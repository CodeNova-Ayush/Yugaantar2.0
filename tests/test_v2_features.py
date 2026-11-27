"""Focused tests for Curriculum v2 encoding (visible_entities) and unlocked-door rewards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from doom_env import CONTRACT_VERSION, make_env
from submission.agent import DoomAgent
from submission.model import (
    INPUT_DIM,
    INPUT_DIM_V2,
    NUM_ACTIONS,
    VITALS_SCALE,
    ActorCritic,
    encode_observation_for_version,
    encode_observation_v2,
)
from training.task_rewards import TaskRewardTracker


def _dummy_obs(with_enemy: bool = True) -> dict:
    feats = np.zeros((32, 5), dtype=np.float32)
    etype = np.zeros(32, dtype=np.int64)
    mask = np.zeros(32, dtype=np.int8)
    if with_enemy:
        feats[0] = [0.25, 0.5, 0.866, 10.0, -5.0]
        etype[0] = 2  # enemy_hitscan
        mask[0] = 1
        feats[1] = [0.4, -0.3, 0.95, 0.0, 0.0]
        etype[1] = 5  # health
        mask[1] = 1
    return {
        "rays": {
            "distance": np.full(64, 0.5, dtype=np.float32),
            "hit_type": np.ones(64, dtype=np.int64),
        },
        "vitals": np.zeros(17, dtype=np.float32),
        "visible_entities": {
            "features": feats,
            "entity_type": etype,
            "mask": mask,
        },
    }


def test_encode_observation_v2_shape_and_values():
    obs = _dummy_obs(with_enemy=True)
    v2 = encode_observation_v2(obs)
    assert v2.shape == (INPUT_DIM_V2,) == (482,)
    assert v2.dtype == np.float32
    # Nearest enemy block starts at INPUT_DIM (465)
    assert v2[INPUT_DIM] == 1.0  # present
    assert np.isclose(v2[INPUT_DIM + 1], 0.25)  # dist
    assert np.isclose(v2[INPUT_DIM + 7], 1.0)  # is_hitscan
    # Nearest pickup block starts at INPUT_DIM + 9
    assert v2[INPUT_DIM + 9] == 1.0
    assert np.isclose(v2[INPUT_DIM + 14], 1.0)  # is_health_or_armor
    # Enemy count / 8
    assert np.isclose(v2[INPUT_DIM + 16], 1.0 / 8.0)


def test_agent_loads_v2_checkpoint_and_preserves_v1_compat(tmp_path: Path):
    env = make_env(Path("maps/two_rooms.wad"), "MAP01", episode_ticks=100, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        # V2 checkpoint
        m2 = ActorCritic(input_dim=INPUT_DIM_V2)
        p2 = tmp_path / "v2.pt"
        torch.save(
            {
                "arch": "mlp",
                "obs_version": "v2",
                "model_state_dict": m2.state_dict(),
                "input_dim": INPUT_DIM_V2,
                "hidden_dim": m2.hidden_dim,
                "num_actions": NUM_ACTIONS,
                "vitals_scale": VITALS_SCALE.tolist(),
                "contract_version": CONTRACT_VERSION,
            },
            p2,
        )
        a2 = DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=p2, seed=0)
        act2 = a2.act(obs)
        assert act2.shape == (9,) and set(np.unique(act2)).issubset({0, 1})

        # V1 checkpoint (without obs_version key, like existing ppo_pilot.pt)
        m1 = ActorCritic(input_dim=INPUT_DIM)
        p1 = tmp_path / "v1.pt"
        torch.save(
            {
                "arch": "mlp",
                "model_state_dict": m1.state_dict(),
                "input_dim": INPUT_DIM,
                "hidden_dim": m1.hidden_dim,
                "num_actions": NUM_ACTIONS,
                "vitals_scale": VITALS_SCALE.tolist(),
                "contract_version": CONTRACT_VERSION,
            },
            p1,
        )
        a1 = DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=p1, seed=0)
        act1 = a1.act(obs)
        assert act1.shape == (9,) and set(np.unique(act1)).issubset({0, 1})
    finally:
        env.close()


def test_task_reward_tracker_unlocked_door_and_reversal():
    s0 = {
        "x": 100.0,
        "y": 100.0,
        "kills": 0,
        "damage_taken": 0.0,
        "door_lock": np.array([-1, 0], dtype=np.int32),
        "sector_floor": np.array([0.0, 0.0]),
        "sector_ceiling": np.array([128.0, 0.0]),
    }
    tr = TaskRewardTracker(s0)
    # Door opens to 40 (not yet passable)
    c1 = tr.step({**s0, "sector_ceiling": np.array([128.0, 40.0])}, unlocked_door_reward=0.4, door_reversal_penalty=0.05)
    assert c1["unlocked_door_bonus"] == 0.0
    assert c1["door_reversal_cost"] == 0.0
    # Door reverses to 32
    c2 = tr.step({**s0, "sector_ceiling": np.array([128.0, 32.0])}, unlocked_door_reward=0.4, door_reversal_penalty=0.05)
    assert c2["door_reversal_cost"] == -0.05
    # Door reaches 56 -> rewarded once
    c3 = tr.step({**s0, "sector_ceiling": np.array([128.0, 56.0])}, unlocked_door_reward=0.4, door_reversal_penalty=0.05)
    assert c3["unlocked_door_bonus"] == 0.4
    # Door stays open -> no repeat reward
    c4 = tr.step({**s0, "sector_ceiling": np.array([128.0, 128.0])}, unlocked_door_reward=0.4, door_reversal_penalty=0.05)
    assert c4["unlocked_door_bonus"] == 0.0
