"""Custom PPO algorithm primitives: config validation, GAE boundary handling, and PPO update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    total_decisions: int = 51200
    rollout_steps: int = 512
    minibatch_size: int = 128
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    exit_reward: float = 1.0
    death_penalty: float = -1.0
    step_cost: float = -0.002
    key_reward: float = 0.0
    door_passable_reward: float = 0.0
    episode_ticks: int = 2100
    kill_reward: float = 0.0
    damage_penalty: float = 0.0
    exploration_reward: float = 0.0
    seed: int = 42
    num_threads: int = 1
    arch: str = "mlp"
    seq_len: int = 64

    def __post_init__(self) -> None:
        if min(self.kill_reward, self.damage_penalty, self.exploration_reward) < 0:
            raise ValueError("Task reward coefficients must be nonnegative")
        if self.arch not in ("mlp", "gru", "key_mem"):
            raise ValueError(f"arch must be 'mlp', 'gru', or 'key_mem', got {self.arch!r}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        if self.key_reward < 0 or self.door_passable_reward < 0:
            raise ValueError("key_reward and door_passable_reward must be >= 0")
        if self.total_decisions <= 0:
            raise ValueError(f"total_decisions must be > 0, got {self.total_decisions}")
        if self.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got {self.rollout_steps}")
        if self.total_decisions % self.rollout_steps != 0:
            raise ValueError(
                f"total_decisions ({self.total_decisions}) must be divisible by rollout_steps ({self.rollout_steps})"
            )
        if self.minibatch_size <= 0 or self.minibatch_size > self.rollout_steps:
            raise ValueError(
                f"minibatch_size ({self.minibatch_size}) must be in [1, rollout_steps ({self.rollout_steps})]"
            )
        if self.update_epochs <= 0:
            raise ValueError(f"update_epochs must be > 0, got {self.update_epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.clip_coef <= 0:
            raise ValueError(f"clip_coef must be > 0, got {self.clip_coef}")
        if self.vf_coef < 0 or self.ent_coef < 0:
            raise ValueError(f"vf_coef ({self.vf_coef}) and ent_coef ({self.ent_coef}) must be >= 0")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        if self.episode_ticks <= 0:
            raise ValueError(f"episode_ticks must be > 0, got {self.episode_ticks}")
        if self.num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {self.num_threads}")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    trunc_bootstrap_values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) with strict episode boundary handling.

    - On termination (exit or death): bootstrap value = 0.0, and do NOT carry advantage across reset.
    - On time-limit truncation: bootstrap value = trunc_bootstrap_values[t] (V(last_obs)),
      and do NOT carry advantage across reset into the new episode.
    - On rollout boundary (t == T - 1, episode still running): bootstrap value = next_value.
    """
    t_len = len(rewards)
    advantages = np.zeros(t_len, dtype=np.float32)
    last_gae_lam = 0.0

    for t in reversed(range(t_len)):
        if terminated[t]:
            next_val = 0.0
            next_nonterminal = 0.0
        elif truncated[t]:
            next_val = float(trunc_bootstrap_values[t])
            next_nonterminal = 0.0
        else:
            next_val = float(next_value) if t == t_len - 1 else float(values[t + 1])
            next_nonterminal = 1.0

        delta = float(rewards[t]) + gamma * next_val - float(values[t])
        last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    old_log_probs_batch: torch.Tensor,
    advantages_batch: torch.Tensor,
    returns_batch: torch.Tensor,
    cfg: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run `cfg.update_epochs` of shuffled-minibatch clipped PPO updates."""
    n = obs_batch.shape[0]
    pg_losses: list[float] = []
    v_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fracs: list[float] = []
    grad_norms: list[float] = []
    valid_counts: list[int] = []
    optimizer_steps = 0

    for _ in range(cfg.update_epochs):
        indices = rng.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            mb_idx = torch.as_tensor(indices[start : start + cfg.minibatch_size], dtype=torch.long)
            valid_counts.append(int(mb_idx.numel()))
            mb_obs = obs_batch[mb_idx]
            mb_actions = actions_batch[mb_idx]
            mb_old_log_probs = old_log_probs_batch[mb_idx]
            mb_advantages = advantages_batch[mb_idx]
            mb_returns = returns_batch[mb_idx]

            if mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            new_log_probs, entropy, new_values = model.evaluate_actions(mb_obs, mb_actions)
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
            ent_loss = entropy.mean()
            loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).item())
            optimizer.step()
            optimizer_steps += 1

            pg_losses.append(float(pg_loss.item()))
            v_losses.append(float(v_loss.item()))
            entropies.append(float(ent_loss.item()))
            approx_kls.append(approx_kl)
            clip_fracs.append(clip_frac)
            grad_norms.append(grad_norm)

    return {
        "policy_loss": float(np.mean(pg_losses)),
        "value_loss": float(np.mean(v_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clip_fracs)),
        "grad_norm": float(np.mean(grad_norms)),
        "optimizer_steps": optimizer_steps,
        "sequence_count": n,
        "mean_valid_transitions_per_minibatch": float(np.mean(valid_counts)),
    }
