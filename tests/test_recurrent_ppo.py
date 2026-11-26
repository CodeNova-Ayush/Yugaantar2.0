"""Tests for compact GRU Actor-Critic, contiguous-sequence recurrent PPO, and DoomAgent parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from doom_env import CONTRACT_VERSION
from doom_env.spaces import action_space, observation_space
from submission.agent import DoomAgent
from submission.model import INPUT_DIM, NUM_ACTIONS, VITALS_SCALE, RecurrentActorCritic, encode_observation
from training.ppo import PPOConfig
from training.recurrent_ppo import build_sequence_chunks, recurrent_ppo_update


def _dummy_obs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = observation_space().sample()
    obs["rays"]["distance"][:] = rng.uniform(0.0, 1.0, 64).astype(np.float32)
    obs["rays"]["hit_type"][:] = rng.integers(0, 6, 64, dtype=np.int64)
    obs["vitals"][:] = rng.uniform(0.0, 50.0, 17).astype(np.float32)
    return obs


def _save_gru_ckpt(path, model: RecurrentActorCritic) -> None:
    torch.save(
        {
            "arch": "gru",
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": model.hidden_dim,
            "gru_hidden_dim": model.gru_hidden_dim,
            "num_actions": NUM_ACTIONS,
            "contract_version": CONTRACT_VERSION,
            "vitals_scale": VITALS_SCALE.tolist(),
            "decisions_done": 512,
        },
        path,
    )


def test_hidden_state_affects_outputs():
    torch.manual_seed(0)
    model = RecurrentActorCritic()
    model.eval()
    x_same = torch.randn(1, INPUT_DIM)
    h0 = model.initial_state(1)
    # Two different step-0 inputs produce different hidden states
    _, _, h_a = model.forward_step(torch.randn(1, INPUT_DIM), h0)
    _, _, h_b = model.forward_step(torch.randn(1, INPUT_DIM), h0)
    assert not torch.allclose(h_a, h_b)
    # Same observation at step 1 with different hidden states produces different logits & values
    logits_a, val_a, _ = model.forward_step(x_same, h_a)
    logits_b, val_b, _ = model.forward_step(x_same, h_b)
    assert not torch.allclose(logits_a, logits_b)
    assert not torch.allclose(val_a, val_b)


def test_step_by_step_matches_sequence_execution():
    torch.manual_seed(1)
    model = RecurrentActorCritic()
    model.eval()
    seq_len = 7
    x_seq = torch.randn(1, seq_len, INPUT_DIM)
    h0 = torch.randn(1, 1, model.gru_hidden_dim)

    seq_logits, seq_values, seq_h_last = model.forward_sequence(x_seq, h0)

    h = h0.clone()
    step_logits = []
    step_values = []
    for t in range(seq_len):
        lt, vt, h = model.forward_step(x_seq[:, t, :], h)
        step_logits.append(lt)
        step_values.append(vt)

    torch.testing.assert_close(torch.stack(step_logits, dim=1), seq_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.stack(step_values, dim=1), seq_values, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h, seq_h_last, atol=1e-5, rtol=1e-5)


def test_agent_reset_removes_previous_episode_influence(tmp_path):
    torch.manual_seed(2)
    model = RecurrentActorCritic()
    ckpt_path = tmp_path / "gru.pt"
    _save_gru_ckpt(ckpt_path, model)

    ep_a = [_dummy_obs(seed=10 + i) for i in range(4)]
    ep_b = [_dummy_obs(seed=20 + i) for i in range(4)]

    fresh = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path, seed=99)
    fresh.reset()
    fresh_actions_b = [fresh.act(o) for o in ep_b]

    reused = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path, seed=99)
    reused.reset()
    for o in ep_a:
        reused.act(o)
    reused.reset()
    reused_actions_b = [reused.act(o) for o in ep_b]

    for a1, a2 in zip(fresh_actions_b, reused_actions_b):
        np.testing.assert_array_equal(a1, a2)


def test_padding_contributes_zero_loss():
    torch.manual_seed(3)
    model_a = RecurrentActorCritic()
    model_b = RecurrentActorCritic()
    model_b.load_state_dict(model_a.state_dict())

    cfg = PPOConfig(total_decisions=512, rollout_steps=512, minibatch_size=128, update_epochs=1, seq_len=8)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg.lr)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg.lr)

    # Create a 6-step rollout with an episode break at t=2 (chunks: [0..2] len 3, [3..5] len 3)
    # Wait: make chunks unequal length: break at t=1 (len 2) and t=5 (len 4) -> max_len=4, chunk 0 has 2 padded steps!
    t_len = 6
    obs = np.random.default_rng(0).normal(size=(t_len, INPUT_DIM)).astype(np.float32)
    actions = np.random.default_rng(1).integers(0, 2, size=(t_len, NUM_ACTIONS)).astype(np.float32)
    log_probs = np.full(t_len, -6.0, dtype=np.float32)
    advantages = np.linspace(-1.0, 1.0, t_len, dtype=np.float32)
    returns = np.linspace(0.0, 1.0, t_len, dtype=np.float32)
    h_in = np.zeros((t_len, model_a.gru_hidden_dim), dtype=np.float32)
    dones = np.array([False, True, False, False, False, True], dtype=bool)

    batch_a = build_sequence_chunks(obs, actions, log_probs, advantages, returns, h_in, dones, seq_len=8)
    batch_b = {k: v.clone() for k, v in batch_a.items()}
    # Corrupt padded entries in chunk 0 (indices 2 and 3 where mask == 0)
    assert batch_b["mask"][0, 2:].sum().item() == 0.0
    batch_b["obs"][0, 2:] = 999.0
    batch_b["actions"][0, 2:] = 1.0
    batch_b["old_log_probs"][0, 2:] = -999.0
    batch_b["advantages"][0, 2:] = 500.0
    batch_b["returns"][0, 2:] = -500.0

    # Use batch_size >= num_chunks by setting minibatch_size=512 so both chunks are in 1 minibatch
    cfg_single = PPOConfig(total_decisions=512, rollout_steps=512, minibatch_size=512, update_epochs=1, seq_len=8)
    metrics_a = recurrent_ppo_update(model_a, opt_a, batch_a, cfg_single, np.random.default_rng(42))
    metrics_b = recurrent_ppo_update(model_b, opt_b, batch_b, cfg_single, np.random.default_rng(42))

    for k in metrics_a:
        assert metrics_a[k] == pytest.approx(metrics_b[k], rel=1e-5, abs=1e-6)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(pa, pb)


def test_gradients_reach_gru():
    torch.manual_seed(4)
    model = RecurrentActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    cfg = PPOConfig(total_decisions=512, rollout_steps=512, minibatch_size=512, update_epochs=1, seq_len=8)

    t_len = 8
    obs = np.random.default_rng(5).normal(size=(t_len, INPUT_DIM)).astype(np.float32)
    actions = np.random.default_rng(6).integers(0, 2, size=(t_len, NUM_ACTIONS)).astype(np.float32)
    log_probs = np.full(t_len, -6.2, dtype=np.float32)
    advantages = np.linspace(-1.0, 1.0, t_len, dtype=np.float32)
    returns = np.linspace(0.0, 1.0, t_len, dtype=np.float32)
    h_in = np.zeros((t_len, model.gru_hidden_dim), dtype=np.float32)
    dones = np.zeros(t_len, dtype=bool)

    batch = build_sequence_chunks(obs, actions, log_probs, advantages, returns, h_in, dones, seq_len=8)
    recurrent_ppo_update(model, optimizer, batch, cfg, np.random.default_rng(0))

    assert model.gru.weight_ih_l0.grad is not None
    assert model.gru.weight_hh_l0.grad is not None
    assert float(model.gru.weight_ih_l0.grad.abs().sum().item()) > 0.0
    assert float(model.gru.weight_hh_l0.grad.abs().sum().item()) > 0.0


def test_gru_checkpoint_reload_reproduces_actions(tmp_path):
    torch.manual_seed(5)
    model = RecurrentActorCritic()
    ckpt_path = tmp_path / "gru_reload.pt"
    _save_gru_ckpt(ckpt_path, model)

    trace = [_dummy_obs(seed=100 + i) for i in range(6)]
    a1 = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path, seed=321)
    a1.reset()
    seq1 = [a1.act(o) for o in trace]

    a2 = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path, seed=321)
    a2.reset()
    seq2 = [a2.act(o) for o in trace]

    for x, y in zip(seq1, seq2):
        np.testing.assert_array_equal(x, y)


def test_sequence_chunking_boundaries_and_initial_hidden_states():
    # 9 steps with seq_len=4:
    # - t=0..1: terminated at t=1 -> chunk 0 has len 2, h0 = h_in[0]
    # - t=2..5: hits seq_len=4 at t=5 (continuing) -> chunk 1 has len 4, h0 = h_in[2]
    # - t=6..7: truncated at t=7 -> chunk 2 has len 2, h0 = h_in[6]
    # - t=8..8: rollout end -> chunk 3 has len 1, h0 = h_in[8]
    t_len = 9
    h_dim = 4
    obs = np.arange(t_len * INPUT_DIM, dtype=np.float32).reshape(t_len, INPUT_DIM)
    actions = np.zeros((t_len, NUM_ACTIONS), dtype=np.float32)
    log_probs = np.zeros(t_len, dtype=np.float32)
    advantages = np.zeros(t_len, dtype=np.float32)
    returns = np.zeros(t_len, dtype=np.float32)
    h_in = np.arange(t_len * h_dim, dtype=np.float32).reshape(t_len, h_dim)
    terminated = np.array([False, True, False, False, False, False, False, False, False], dtype=bool)
    truncated = np.array([False, False, False, False, False, False, False, True, False], dtype=bool)
    dones = terminated | truncated

    batch = build_sequence_chunks(obs, actions, log_probs, advantages, returns, h_in, dones, seq_len=4)
    assert batch["obs"].shape == (4, 4, INPUT_DIM)
    assert batch["mask"].tolist() == [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    np.testing.assert_array_equal(batch["h0"][0, 0].numpy(), h_in[0])
    np.testing.assert_array_equal(batch["h0"][0, 1].numpy(), h_in[2])
    np.testing.assert_array_equal(batch["h0"][0, 2].numpy(), h_in[6])
    np.testing.assert_array_equal(batch["h0"][0, 3].numpy(), h_in[8])
