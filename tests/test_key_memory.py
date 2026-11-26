"""Unit tests for KeyMemoryActorCritic, recurrent PPO with memory slots, and DoomAgent parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from doom_env import CONTRACT_VERSION
from doom_env.spaces import action_space, observation_space
from submission.agent import DoomAgent
from submission.model import INPUT_DIM, NUM_ACTIONS, VITALS_SCALE, KeyMemoryActorCritic
from training.ppo import PPOConfig
from training.recurrent_ppo import build_sequence_chunks, recurrent_ppo_update


def _dummy_obs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = observation_space().sample()
    obs["rays"]["distance"][:] = rng.uniform(0.0, 1.0, 64).astype(np.float32)
    obs["rays"]["hit_type"][:] = rng.integers(0, 6, 64, dtype=np.int64)
    obs["vitals"][:] = rng.uniform(0.0, 50.0, 17).astype(np.float32)
    obs["vitals"][8:14] = rng.integers(0, 2, 6).astype(np.float32)
    return obs


def _save_key_mem_ckpt(path, model: KeyMemoryActorCritic) -> None:
    torch.save(
        {
            "arch": "key_mem",
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": model.hidden_dim,
            "gru_hidden_dim": model.gru_hidden_dim,
            "num_slots": model.num_slots,
            "slot_dim": model.slot_dim,
            "num_actions": NUM_ACTIONS,
            "contract_version": CONTRACT_VERSION,
            "vitals_scale": VITALS_SCALE.tolist(),
            "decisions_done": 512,
        },
        path,
    )


def test_key_mem_step_matches_sequence():
    torch.manual_seed(0)
    model = KeyMemoryActorCritic()
    model.eval()
    x_seq = torch.randn(1, 6, INPUT_DIM)
    h0 = torch.randn(1, 1, model.gru_hidden_dim)
    mem0 = torch.randn(1, model.num_slots, model.slot_dim)

    seq_logits, seq_values, (seq_h, seq_mem) = model.forward_sequence(x_seq, (h0, mem0))

    state = (h0.clone(), mem0.clone())
    step_logits, step_values = [], []
    for t in range(6):
        lt, vt, state = model.forward_step(x_seq[:, t, :], state)
        step_logits.append(lt)
        step_values.append(vt)

    torch.testing.assert_close(torch.stack(step_logits, dim=1), seq_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.stack(step_values, dim=1), seq_values, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state[0], seq_h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state[1], seq_mem, atol=1e-5, rtol=1e-5)


def test_key_mem_reset_isolation_and_checkpoint_reload(tmp_path):
    torch.manual_seed(1)
    model = KeyMemoryActorCritic()
    ckpt_path = tmp_path / "key_mem.pt"
    _save_key_mem_ckpt(ckpt_path, model)

    ep_a = [_dummy_obs(10 + i) for i in range(4)]
    ep_b = [_dummy_obs(20 + i) for i in range(4)]

    fresh = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path, seed=77)
    fresh.reset()
    acts_fresh = [fresh.act(o) for o in ep_b]

    reused = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path, seed=77)
    reused.reset()
    for o in ep_a:
        reused.act(o)
    reused.reset()
    acts_reused = [reused.act(o) for o in ep_b]

    for a1, a2 in zip(acts_fresh, acts_reused):
        np.testing.assert_array_equal(a1, a2)


def test_key_mem_padding_zero_loss_and_gradients_reach_memory():
    torch.manual_seed(2)
    model_a = KeyMemoryActorCritic()
    model_b = KeyMemoryActorCritic()
    model_b.load_state_dict(model_a.state_dict())

    cfg = PPOConfig(total_decisions=512, rollout_steps=512, minibatch_size=512, update_epochs=1, seq_len=8, arch="key_mem")
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg.lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg.lr)

    t_len = 6
    obs = np.random.default_rng(0).normal(size=(t_len, INPUT_DIM)).astype(np.float32)
    actions = np.random.default_rng(1).integers(0, 2, size=(t_len, NUM_ACTIONS)).astype(np.float32)
    log_probs = np.full(t_len, -6.0, dtype=np.float32)
    advantages = np.linspace(-1.0, 1.0, t_len, dtype=np.float32)
    returns = np.linspace(0.0, 1.0, t_len, dtype=np.float32)
    h_in = np.zeros((t_len, model_a.gru_hidden_dim), dtype=np.float32)
    mem_in = np.zeros((t_len, model_a.num_slots, model_a.slot_dim), dtype=np.float32)
    dones = np.array([False, True, False, False, False, True], dtype=bool)

    batch_a = build_sequence_chunks(obs, actions, log_probs, advantages, returns, h_in, dones, seq_len=8, mem_in=mem_in)
    batch_b = {k: v.clone() for k, v in batch_a.items()}
    assert batch_b["mask"][0, 2:].sum().item() == 0.0
    batch_b["obs"][0, 2:] = 999.0
    batch_b["old_log_probs"][0, 2:] = -999.0
    batch_b["advantages"][0, 2:] = 500.0
    batch_b["returns"][0, 2:] = -500.0

    m_a = recurrent_ppo_update(model_a, opt_a, batch_a, cfg, np.random.default_rng(42))
    m_b = recurrent_ppo_update(model_b, opt_b, batch_b, cfg, np.random.default_rng(42))

    for k in m_a:
        assert m_a[k] == pytest.approx(m_b[k], rel=1e-5, abs=1e-6)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(pa, pb)

    # Verify gradients reached memory read/write parameters
    for param in (
        model_a.write_gate.weight,
        model_a.write_proj.weight,
        model_a.read_query.weight,
        model_a.gru.weight_ih_l0,
    ):
        assert param.grad is not None and float(param.grad.abs().sum().item()) > 0.0
