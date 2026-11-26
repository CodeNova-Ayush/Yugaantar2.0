"""Shared observation preprocessing and neural Actor-Critic policies (MLP and GRU).

Used by both training (training/train_ppo.py) and submission inference
(submission/agent.py) so inputs and action distributions match exactly.

Input encoding (465 features total):
  - 64 ray distances in [0, 1] (already normalized by RAY_RANGE = 1024.0)
  - 384 ray hit-type one-hot features (64 rays x 6 categories)
  - 17 scaled vitals:
      * health / 100.0, armor / 100.0
      * selected_weapon / 7.0
      * selected_weapon_ammo, ammo_bullets, ammo_shells, ammo_rockets, ammo_cells / 50.0
      * 6 binary key flags preserved at 1.0 (0 or 1)
      * 3 signed velocities (vel_forward, vel_right, vel_up) / 20.0 (sign preserved)

No coordinates, map identity, privileged state, or scripted steering are used.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from doom_env.spaces import BUTTONS, NUM_RAYS, RAY_HIT_TYPES, VITALS_FIELDS

NUM_RAY_HIT_TYPES = len(RAY_HIT_TYPES)  # 6
NUM_VITALS = len(VITALS_FIELDS)  # 17
NUM_ACTIONS = len(BUTTONS)  # 9
INPUT_DIM = NUM_RAYS + NUM_RAYS * NUM_RAY_HIT_TYPES + NUM_VITALS  # 64 + 384 + 17 = 465
# Indices of the 6 official binary key flags inside the 465-d encoded observation.
KEY_FLAGS_START = NUM_RAYS + NUM_RAYS * NUM_RAY_HIT_TYPES + 8
KEY_FLAGS_END = KEY_FLAGS_START + 6

# Fixed per-vital scale factors (documented above; binary keys = 1.0, velocities = 20.0).
VITALS_SCALE = np.array(
    [
        100.0,  # health
        100.0,  # armor
        7.0,    # selected_weapon
        50.0,   # selected_weapon_ammo
        50.0,   # ammo_bullets
        50.0,   # ammo_shells
        50.0,   # ammo_rockets
        50.0,   # ammo_cells
        1.0,    # key_blue_card (binary)
        1.0,    # key_yellow_card (binary)
        1.0,    # key_red_card (binary)
        1.0,    # key_blue_skull (binary)
        1.0,    # key_yellow_skull (binary)
        1.0,    # key_red_skull (binary)
        20.0,   # vel_forward (signed)
        20.0,   # vel_right (signed)
        20.0,   # vel_up (signed)
    ],
    dtype=np.float32,
)


def encode_observation(obs: dict) -> np.ndarray:
    """Encode a single doom_env observation dict into a 465-element float32 vector."""
    distances = np.asarray(obs["rays"]["distance"], dtype=np.float32)
    hit_types = np.asarray(obs["rays"]["hit_type"], dtype=np.int64)
    one_hot = np.eye(NUM_RAY_HIT_TYPES, dtype=np.float32)[hit_types].reshape(-1)
    vitals = np.asarray(obs["vitals"], dtype=np.float32) / VITALS_SCALE
    vec = np.concatenate([distances, one_hot, vitals], axis=0)
    return vec.astype(np.float32, copy=False)


class ActorCritic(nn.Module):
    """Minimal feedforward MLP with a 9-logit Bernoulli policy head and 1 scalar value head."""

    arch = "mlp"

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 128, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (9 policy logits, scalar state value) for a batch of encoded observations."""
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def evaluate_actions(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint log_prob (summed over 9 buttons), entropy (summed), and value."""
        logits, value = self.forward(x)
        dist = Bernoulli(logits=logits)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def sample_action(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a 9-button binary action and return (action, joint_log_prob, value)."""
        logits, value = self.forward(x)
        probs = torch.sigmoid(logits)
        sample = torch.bernoulli(probs, generator=generator)
        dist = Bernoulli(logits=logits)
        log_prob = dist.log_prob(sample).sum(dim=-1)
        return sample, log_prob, value


class RecurrentActorCritic(nn.Module):
    """Compact GRU Actor-Critic with the same 465-d input and 9-button Bernoulli output."""

    arch = "gru"

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(gru_hidden_dim, num_actions)
        self.value_head = nn.Linear(gru_hidden_dim, 1)

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return zeroed GRU hidden state of shape (1, batch_size, gru_hidden_dim)."""
        return torch.zeros(1, batch_size, self.gru_hidden_dim, dtype=torch.float32, device=device)

    def forward_step(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward: x (B, 465), h (1, B, H) -> (logits (B, 9), value (B,), next_h (1, B, H))."""
        enc = self.encoder(x).unsqueeze(1)  # (B, 1, hidden_dim)
        out, next_h = self.gru(enc, h)
        feat = out.squeeze(1)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value, next_h

    def forward_sequence(
        self, x_seq: torch.Tensor, h0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence forward: x_seq (B, T, 465), h0 (1, B, H) -> (logits (B, T, 9), values (B, T), h_last)."""
        enc = self.encoder(x_seq)
        out, h_last = self.gru(enc, h0)
        logits = self.policy_head(out)
        values = self.value_head(out).squeeze(-1)
        return logits, values, h_last

    def evaluate_sequences(
        self, x_seq: torch.Tensor, h0: torch.Tensor, actions_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate joint log_prob (B, T), entropy (B, T), and value (B, T) over sequences."""
        logits, values, _ = self.forward_sequence(x_seq, h0)
        dist = Bernoulli(logits=logits)
        log_prob = dist.log_prob(actions_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, values

    @torch.no_grad()
    def sample_step(
        self, x: torch.Tensor, h: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a 9-button binary action for one step and return (action, log_prob, value, next_h)."""
        logits, value, next_h = self.forward_step(x, h)
        probs = torch.sigmoid(logits)
        sample = torch.bernoulli(probs, generator=generator)
        dist = Bernoulli(logits=logits)
        log_prob = dist.log_prob(sample).sum(dim=-1)
        return sample, log_prob, value, next_h


class KeyMemoryActorCritic(nn.Module):
    """GRU Actor-Critic with a small learned slot memory and key-conditioned read attention.

    Memory slots are updated via learned write gates from the GRU state and read
    via dot-product attention conditioned on (GRU state, 6 official key flags).
    """

    arch = "key_mem"

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
        num_slots: int = 4,
        slot_dim: int = 32,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.write_gate = nn.Linear(gru_hidden_dim, num_slots)
        self.write_proj = nn.Linear(gru_hidden_dim, num_slots * slot_dim)
        self.read_query = nn.Linear(gru_hidden_dim + 6, slot_dim)
        self.policy_head = nn.Linear(gru_hidden_dim + slot_dim, num_actions)
        self.value_head = nn.Linear(gru_hidden_dim + slot_dim, 1)

    def initial_state(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed (gru_h (1, B, H), memory (B, num_slots, slot_dim))."""
        h = torch.zeros(1, batch_size, self.gru_hidden_dim, dtype=torch.float32, device=device)
        mem = torch.zeros(batch_size, self.num_slots, self.slot_dim, dtype=torch.float32, device=device)
        return h, mem

    def _memory_step(
        self, g: torch.Tensor, keys: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One learned write + key-conditioned read step: g (B, H), keys (B, 6), mem (B, S, D)."""
        bsz = g.shape[0]
        gate = torch.sigmoid(self.write_gate(g)).unsqueeze(-1)  # (B, S, 1)
        cand = torch.tanh(self.write_proj(g)).view(bsz, self.num_slots, self.slot_dim)
        mem_next = (1.0 - gate) * mem + gate * cand

        query = self.read_query(torch.cat([g, keys], dim=-1))  # (B, D)
        scores = (mem_next * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.slot_dim)  # (B, S)
        attn = torch.softmax(scores, dim=-1)
        read_vec = (attn.unsqueeze(-1) * mem_next).sum(dim=1)  # (B, D)
        return read_vec, mem_next

    def forward_step(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step forward: x (B, 465), state=(h, mem) -> (logits, value, next_state)."""
        h, mem = state
        enc = self.encoder(x).unsqueeze(1)
        out, next_h = self.gru(enc, h)
        g = out.squeeze(1)
        keys = x[:, KEY_FLAGS_START:KEY_FLAGS_END]
        read_vec, next_mem = self._memory_step(g, keys, mem)
        feat = torch.cat([g, read_vec], dim=-1)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value, (next_h, next_mem)

    def forward_sequence(
        self, x_seq: torch.Tensor, state0: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Sequence forward: x_seq (B, T, 465), state0=(h0, mem0) -> (logits, values, state_last)."""
        h0, mem = state0
        enc = self.encoder(x_seq)
        g_seq, h_last = self.gru(enc, h0)
        keys_seq = x_seq[:, :, KEY_FLAGS_START:KEY_FLAGS_END]

        reads: list[torch.Tensor] = []
        for t in range(x_seq.shape[1]):
            read_t, mem = self._memory_step(g_seq[:, t, :], keys_seq[:, t, :], mem)
            reads.append(read_t)
        read_seq = torch.stack(reads, dim=1)
        feat_seq = torch.cat([g_seq, read_seq], dim=-1)
        logits = self.policy_head(feat_seq)
        values = self.value_head(feat_seq).squeeze(-1)
        return logits, values, (h_last, mem)

    def evaluate_sequences(
        self, x_seq: torch.Tensor, state0: tuple[torch.Tensor, torch.Tensor], actions_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate joint log_prob (B, T), entropy (B, T), and value (B, T) over sequences."""
        logits, values, _ = self.forward_sequence(x_seq, state0)
        dist = Bernoulli(logits=logits)
        log_prob = dist.log_prob(actions_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, values

    @torch.no_grad()
    def sample_step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Sample a 9-button binary action for one step and return (action, log_prob, value, next_state)."""
        logits, value, next_state = self.forward_step(x, state)
        probs = torch.sigmoid(logits)
        sample = torch.bernoulli(probs, generator=generator)
        dist = Bernoulli(logits=logits)
        log_prob = dist.log_prob(sample).sum(dim=-1)
        return sample, log_prob, value, next_state


def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    """Instantiate ActorCritic, RecurrentActorCritic, or KeyMemoryActorCritic from metadata."""
    arch = str(ckpt.get("arch", "mlp"))
    input_dim = int(ckpt["input_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    num_actions = int(ckpt["num_actions"])
    if arch == "mlp":
        return ActorCritic(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)
    if arch == "gru":
        if "gru_hidden_dim" not in ckpt:
            raise ValueError("Incompatible GRU checkpoint: missing 'gru_hidden_dim'")
        return RecurrentActorCritic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=int(ckpt["gru_hidden_dim"]),
            num_actions=num_actions,
        )
    if arch == "key_mem":
        for k in ("gru_hidden_dim", "num_slots", "slot_dim"):
            if k not in ckpt:
                raise ValueError(f"Incompatible key_mem checkpoint: missing {k!r}")
        return KeyMemoryActorCritic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=int(ckpt["gru_hidden_dim"]),
            num_slots=int(ckpt["num_slots"]),
            slot_dim=int(ckpt["slot_dim"]),
            num_actions=num_actions,
        )
    raise ValueError(f"Unsupported checkpoint architecture: {arch!r}")
