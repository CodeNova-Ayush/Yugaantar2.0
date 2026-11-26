"""Contiguous-sequence recurrent PPO (chunked at episode and TBPTT boundaries).

Stored-hidden-state approximation:
  Each sequence chunk preserves its initial hidden state `h0 = h_in[start]`
  recorded during rollout collection under the pre-update policy. Across
  multiple epochs/minibatches within a PPO update, `h0` is held fixed rather
  than recomputed with updated weights (standard PPO recurrent approximation).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training.ppo import PPOConfig


def build_sequence_chunks(
    obs: np.ndarray,
    actions: np.ndarray,
    log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    h_in: np.ndarray,
    dones: np.ndarray,
    seq_len: int,
    mem_in: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Split a rollout into contiguous within-episode sequences of length <= seq_len.

    Sequences break at both episode boundaries (dones[t] == True) and TBPTT chunk
    boundaries (length == seq_len). Each sequence preserves its exact initial hidden
    state h0 = h_in[start] and pads shorter sequences with mask = 0.0.
    """
    t_len = len(obs)
    chunks: list[tuple[int, int]] = []
    start = 0
    for t in range(t_len):
        cur_len = t - start + 1
        if dones[t] or cur_len == seq_len or t == t_len - 1:
            chunks.append((start, t + 1))
            start = t + 1

    n_seq = len(chunks)
    max_len = max(end - s for s, end in chunks)
    obs_dim = obs.shape[1]
    act_dim = actions.shape[1]
    h_dim = h_in.shape[1]

    seq_obs = np.zeros((n_seq, max_len, obs_dim), dtype=np.float32)
    seq_actions = np.zeros((n_seq, max_len, act_dim), dtype=np.float32)
    seq_log_probs = np.zeros((n_seq, max_len), dtype=np.float32)
    seq_advantages = np.zeros((n_seq, max_len), dtype=np.float32)
    seq_returns = np.zeros((n_seq, max_len), dtype=np.float32)
    seq_mask = np.zeros((n_seq, max_len), dtype=np.float32)
    seq_h0 = np.zeros((1, n_seq, h_dim), dtype=np.float32)
    seq_mem0 = (
        np.zeros((n_seq, mem_in.shape[1], mem_in.shape[2]), dtype=np.float32)
        if mem_in is not None
        else None
    )

    for i, (s, end) in enumerate(chunks):
        length = end - s
        seq_obs[i, :length] = obs[s:end]
        seq_actions[i, :length] = actions[s:end]
        seq_log_probs[i, :length] = log_probs[s:end]
        seq_advantages[i, :length] = advantages[s:end]
        seq_returns[i, :length] = returns[s:end]
        seq_mask[i, :length] = 1.0
        seq_h0[0, i] = h_in[s]
        if seq_mem0 is not None and mem_in is not None:
            seq_mem0[i] = mem_in[s]

    out = {
        "obs": torch.from_numpy(seq_obs),
        "actions": torch.from_numpy(seq_actions),
        "old_log_probs": torch.from_numpy(seq_log_probs),
        "advantages": torch.from_numpy(seq_advantages),
        "returns": torch.from_numpy(seq_returns),
        "mask": torch.from_numpy(seq_mask),
        "h0": torch.from_numpy(seq_h0),
    }
    if seq_mem0 is not None:
        out["mem0"] = torch.from_numpy(seq_mem0)
    return out


def recurrent_ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    seq_batch: dict[str, torch.Tensor],
    cfg: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run `cfg.update_epochs` of sequence-shuffled, masked clipped PPO updates."""
    n_seq = seq_batch["obs"].shape[0]
    seq_mb_size = max(1, cfg.minibatch_size // cfg.seq_len)

    pg_losses: list[float] = []
    v_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fracs: list[float] = []
    grad_norms: list[float] = []
    valid_counts: list[float] = []
    optimizer_steps = 0

    for _ in range(cfg.update_epochs):
        perm = rng.permutation(n_seq)
        for start in range(0, n_seq, seq_mb_size):
            idx = torch.as_tensor(perm[start : start + seq_mb_size], dtype=torch.long)
            mb_obs = seq_batch["obs"][idx]
            mb_actions = seq_batch["actions"][idx]
            mb_old_lp = seq_batch["old_log_probs"][idx]
            mb_adv = seq_batch["advantages"][idx]
            mb_ret = seq_batch["returns"][idx]
            mb_mask = seq_batch["mask"][idx]
            mb_h0 = seq_batch["h0"][:, idx, :].detach()
            state0 = (mb_h0, seq_batch["mem0"][idx].detach()) if "mem0" in seq_batch else mb_h0

            valid_count = mb_mask.sum().clamp_min(1.0)
            valid_counts.append(float(mb_mask.sum().item()))
            valid_adv = mb_adv[mb_mask > 0.5]
            if valid_adv.numel() > 1:
                adv_mean = valid_adv.mean()
                adv_std = valid_adv.std()
                mb_adv = (mb_adv - adv_mean) / (adv_std + 1e-8)

            new_lp, entropy, new_val = model.evaluate_sequences(mb_obs, state0, mb_actions)
            log_ratio = (new_lp - mb_old_lp) * mb_mask
            ratio = torch.exp(log_ratio)

            with torch.no_grad():
                approx_kl = ((((ratio - 1.0) - log_ratio) * mb_mask).sum() / valid_count).item()
                clip_frac = ((((ratio - 1.0).abs() > cfg.clip_coef).float() * mb_mask).sum() / valid_count).item()

            mb_adv_masked = mb_adv * mb_mask
            pg_loss1 = -mb_adv_masked * ratio
            pg_loss2 = -mb_adv_masked * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_mask).sum() / valid_count

            v_err = (new_val - mb_ret) * mb_mask
            v_loss = 0.5 * (v_err ** 2).sum() / valid_count
            ent_loss = (entropy * mb_mask).sum() / valid_count
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
        "sequence_count": n_seq,
        "mean_valid_transitions_per_minibatch": float(np.mean(valid_counts)),
    }
