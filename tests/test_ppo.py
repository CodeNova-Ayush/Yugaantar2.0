"""Focused tests for preprocessing, joint Bernoulli log-probs, GAE, config/checkpoint integrity, and DoomAgent parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli

from doom_env import CONTRACT_VERSION
from doom_env.spaces import action_space, observation_space
from submission.agent import DoomAgent
from submission.model import INPUT_DIM, NUM_ACTIONS, VITALS_SCALE, ActorCritic, encode_observation
from training.curriculum.verify_curriculum import canonical_d4_signature, check_split_hygiene
from training.ppo import PPOConfig, compute_gae
from training.train_ppo import ensure_fresh_dir, train


def _dummy_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs = observation_space().sample()
    obs["rays"]["distance"][:] = rng.uniform(0.0, 1.0, 64).astype(np.float32)
    obs["rays"]["hit_type"][:] = rng.integers(0, 6, 64, dtype=np.int64)
    obs["vitals"][:] = 0.0
    obs["vitals"][0] = 100.0
    obs["vitals"][8] = 1.0
    obs["vitals"][13] = 0.0
    obs["vitals"][14] = -10.0
    obs["vitals"][15] = 20.0
    return obs


def _valid_ckpt(model: ActorCritic, **overrides) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "hidden_dim": 128,
        "num_actions": NUM_ACTIONS,
        "contract_version": CONTRACT_VERSION,
        "vitals_scale": VITALS_SCALE.tolist(),
        "decisions_done": 512,
    }
    payload.update(overrides)
    return payload


def test_encode_observation_shape_and_scaling():
    obs = _dummy_obs()
    vec = encode_observation(obs)
    assert vec.shape == (INPUT_DIM,) == (465,)
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec[:64], obs["rays"]["distance"])
    vitals = vec[64 + 384 :]
    assert vitals[0] == pytest.approx(1.0)
    assert vitals[8] == pytest.approx(1.0)
    assert vitals[13] == pytest.approx(0.0)
    assert vitals[14] == pytest.approx(-0.5)
    assert vitals[15] == pytest.approx(1.0)


def test_joint_log_prob_and_entropy_sum_across_nine_buttons():
    model = ActorCritic()
    x = torch.randn(4, INPUT_DIM)
    actions = torch.randint(0, 2, (4, NUM_ACTIONS), dtype=torch.float32)

    logits, _ = model(x)
    dist = Bernoulli(logits=logits)
    expected_log_prob = dist.log_prob(actions).sum(dim=-1)
    expected_entropy = dist.entropy().sum(dim=-1)

    log_prob, entropy, _ = model.evaluate_actions(x, actions)
    torch.testing.assert_close(log_prob, expected_log_prob)
    torch.testing.assert_close(entropy, expected_entropy)


def test_gae_boundary_handling():
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 1.0, 1.5], dtype=np.float32)
    terminated = np.array([False, True, False, False], dtype=bool)
    truncated = np.array([False, False, True, False], dtype=bool)
    trunc_boot = np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float32)
    next_value = 2.0
    gamma, lam = 0.99, 0.95

    adv, ret = compute_gae(rewards, values, terminated, truncated, trunc_boot, next_value, gamma, lam)
    adv3 = 4.0 + gamma * next_value - values[3]
    adv2 = 3.0 + gamma * 10.0 - values[2]
    adv1 = 2.0 + 0.0 - values[1]
    adv0 = (1.0 + gamma * values[1] - values[0]) + gamma * lam * adv1

    np.testing.assert_allclose(adv, [adv0, adv1, adv2, adv3], rtol=1e-5)
    np.testing.assert_allclose(ret, adv + values, rtol=1e-5)


def test_config_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="divisible"):
        PPOConfig(total_decisions=1000, rollout_steps=512)
    with pytest.raises(ValueError, match="minibatch_size"):
        PPOConfig(total_decisions=512, rollout_steps=512, minibatch_size=1024)
    with pytest.raises(ValueError, match="lr"):
        PPOConfig(lr=0.0)
    with pytest.raises(ValueError, match="gamma"):
        PPOConfig(gamma=1.5)


def test_refuses_overwrite_existing_dir(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "summary.json").write_text("{}")
    with pytest.raises(FileExistsError):
        ensure_fresh_dir(exp_dir, overwrite=False)
    ensure_fresh_dir(exp_dir, overwrite=True)


def test_checkpoint_validation_and_doom_agent_parity(tmp_path):
    model = ActorCritic()
    ckpt_path = tmp_path / "valid.pt"
    torch.save(_valid_ckpt(model), ckpt_path)

    # Multi-observation parity between DoomAgent and ActorCritic.sample_action
    observations = [_dummy_obs(seed=i) for i in range(5)]
    agent = DoomAgent(observation_space(), action_space(), "cpu", weights_path=ckpt_path, seed=777)
    agent.reset()
    agent_actions = [agent.act(obs) for obs in observations]

    gen = torch.Generator(device="cpu").manual_seed(777)
    model.eval()
    direct_actions = []
    for obs in observations:
        x = torch.from_numpy(encode_observation(obs)).unsqueeze(0)
        act_t, _, _ = model.sample_action(x, generator=gen)
        direct_actions.append(act_t.squeeze(0).numpy().astype(np.int8))

    for a_ag, a_dir in zip(agent_actions, direct_actions):
        np.testing.assert_array_equal(a_ag, a_dir)

    # Reject missing, wrong contract_version, and wrong vitals_scale
    with pytest.raises(FileNotFoundError):
        DoomAgent(observation_space(), action_space(), "cpu", weights_path=tmp_path / "missing.pt")

    bad_ver = tmp_path / "bad_ver.pt"
    torch.save(_valid_ckpt(model, contract_version="0.0.0-bad"), bad_ver)
    with pytest.raises(ValueError, match="contract_version"):
        DoomAgent(observation_space(), action_space(), "cpu", weights_path=bad_ver)

    bad_scale = tmp_path / "bad_scale.pt"
    torch.save(_valid_ckpt(model, vitals_scale=[1.0] * 17), bad_scale)
    with pytest.raises(ValueError, match="vitals_scale"):
        DoomAgent(observation_space(), action_space(), "cpu", weights_path=bad_scale)


def test_d4_canonicalization_ignores_start_and_exit_labels():
    grid_a = [
        "#####",
        "#P..#",
        "#..E#",
        "#####",
    ]
    # Same walkable geometry, different P and E positions, rotated 90 degrees
    grid_b = [
        "####",
        "#E.#",
        "#..#",
        "#.P#",
        "####",
    ]
    assert canonical_d4_signature(grid_a) == canonical_d4_signature(grid_b)
    with pytest.raises(AssertionError, match="D4 geometry overlap"):
        check_split_hygiene(
            [
                {"name": "m1", "split": "train", "grid": grid_a},
                {"name": "m2", "split": "test", "grid": grid_b},
            ]
        )


def test_resumable_checkpoint_roundtrip(tmp_path):
    manifest = tmp_path / "manifest.json"
    # Use the existing practice_corridor WAD for a fast 2-update resume test
    from pathlib import Path
    import json
    orig = json.loads(Path("training/progression/manifest.json").read_text())
    one_train = [m for m in orig["maps"] if m["split"] == "train"][:1]
    manifest.write_text(json.dumps({"maps": one_train}))

    out_dir = tmp_path / "resumable_run"
    cfg1 = PPOConfig(total_decisions=512, rollout_steps=512, minibatch_size=128, update_epochs=1, seed=7)
    s1 = train(cfg1, manifest, out_dir, save_weights=None, checkpoint_every=512)
    assert s1["decisions_done"] == 512
    ts_path = out_dir / "checkpoints" / "train_state_512.pt"
    assert ts_path.is_file()

    cfg2 = PPOConfig(total_decisions=1024, rollout_steps=512, minibatch_size=128, update_epochs=1, seed=7)
    s2 = train(cfg2, manifest, out_dir, save_weights=None, checkpoint_every=512, resume_from=ts_path)
    assert s2["decisions_done"] == 1024
    assert s2["final_update"]["update"] == 2
    assert (out_dir / "checkpoints" / "ckpt_1024.pt").is_file()

    # Incompatible rollout_steps, arch, lr, seed, and map hash must be rejected
    with pytest.raises(ValueError, match="rollout_steps"):
        train(PPOConfig(total_decisions=1024, rollout_steps=256, minibatch_size=128, update_epochs=1, seed=7), manifest, out_dir, None, resume_from=ts_path)
    with pytest.raises(ValueError, match="arch"):
        train(PPOConfig(total_decisions=1024, rollout_steps=512, minibatch_size=128, update_epochs=1, seed=7, arch="gru"), manifest, out_dir, None, resume_from=ts_path)
    with pytest.raises(ValueError, match="lr"):
        train(PPOConfig(total_decisions=1024, rollout_steps=512, minibatch_size=128, update_epochs=1, lr=1e-4, seed=7), manifest, out_dir, None, resume_from=ts_path)
    with pytest.raises(ValueError, match="seed"):
        train(PPOConfig(total_decisions=1024, rollout_steps=512, minibatch_size=128, update_epochs=1, seed=99), manifest, out_dir, None, resume_from=ts_path)

    bad_manifest = tmp_path / "bad_manifest.json"
    bad_train = [{**one_train[0], "sha256": "0" * 64}]
    bad_manifest.write_text(json.dumps({"maps": bad_train}))
    with pytest.raises((ValueError, AssertionError)):
        train(cfg2, bad_manifest, out_dir, None, resume_from=ts_path)
