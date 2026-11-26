"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
Loads trained Actor-Critic weights (MLP or GRU) from submission/weights/
and selects actions via stochastic Bernoulli sampling over all 9 buttons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from doom_env import CONTRACT_VERSION
from submission.model import (
    INPUT_DIM,
    NUM_ACTIONS,
    VITALS_SCALE,
    KeyMemoryActorCritic,
    RecurrentActorCritic,
    build_model_from_checkpoint,
    encode_observation,
)

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "ppo_pilot.pt"


class DoomAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        device: str,
        weights_path: Path | None = None,
        seed: int = 0,
        clear_memory_each_step: bool = False,
    ):
        """Called once. Load weights from WEIGHTS_DIR onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.seed = int(seed)
        if self.device.type == "cpu":
            torch.set_num_threads(1)

        path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing trained weights at {path}. Train or place ppo_pilot.pt in submission/weights/."
            )

        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        required_keys = {
            "model_state_dict",
            "input_dim",
            "hidden_dim",
            "num_actions",
            "contract_version",
            "vitals_scale",
        }
        missing = required_keys - set(ckpt.keys())
        if missing:
            raise ValueError(f"Incompatible checkpoint {path}: missing keys {sorted(missing)}")
        if str(ckpt["contract_version"]) != CONTRACT_VERSION:
            raise ValueError(
                f"Incompatible checkpoint {path}: contract_version={ckpt['contract_version']!r} "
                f"!= {CONTRACT_VERSION!r}"
            )
        if int(ckpt["input_dim"]) != INPUT_DIM or int(ckpt["num_actions"]) != NUM_ACTIONS:
            raise ValueError(
                f"Incompatible checkpoint {path}: expected input_dim={INPUT_DIM}, num_actions={NUM_ACTIONS}, "
                f"got input_dim={ckpt['input_dim']}, num_actions={ckpt['num_actions']}"
            )
        ckpt_scale = np.asarray(ckpt["vitals_scale"], dtype=np.float32)
        if ckpt_scale.shape != VITALS_SCALE.shape or not np.allclose(ckpt_scale, VITALS_SCALE):
            raise ValueError(f"Incompatible checkpoint {path}: vitals_scale mismatch")
        if tuple(action_space.shape) != (NUM_ACTIONS,):
            raise ValueError(f"Expected action_space shape ({NUM_ACTIONS},), got {action_space.shape}")

        self.model = build_model_from_checkpoint(ckpt).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.is_recurrent = isinstance(self.model, (RecurrentActorCritic, KeyMemoryActorCritic))
        self.is_key_mem = isinstance(self.model, KeyMemoryActorCritic)
        self.clear_memory_each_step = bool(clear_memory_each_step)
        self.hidden_state = self.model.initial_state(1, self.device) if self.is_recurrent else None
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def reset(self) -> None:
        """Called before every map. Reset recurrent hidden state (if GRU/key_mem) and RNG."""
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if self.is_recurrent:
            self.hidden_state = self.model.initial_state(1, self.device)

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        x = torch.from_numpy(encode_observation(observation)).unsqueeze(0).to(self.device)
        if self.is_recurrent:
            if self.is_key_mem and self.clear_memory_each_step:
                h, mem = self.hidden_state
                self.hidden_state = (h, torch.zeros_like(mem))
            action_t, _, _, self.hidden_state = self.model.sample_step(
                x, self.hidden_state, generator=self.generator
            )
        else:
            action_t, _, _ = self.model.sample_action(x, generator=self.generator)
        return action_t.squeeze(0).cpu().numpy().astype(np.int8)
