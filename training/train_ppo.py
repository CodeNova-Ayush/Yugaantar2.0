"""Train the feedforward (MLP) or recurrent (GRU) Actor-Critic policy with custom PPO.

Trains ONLY on maps in the training split of the supplied manifest.
Never touches validation or test maps during training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import CONTRACT_VERSION, make_env  # noqa: E402
from doom_env.spaces import FRAME_SKIP, VITALS_INDEX  # noqa: E402
from submission.model import (  # noqa: E402
    INPUT_DIM,
    NUM_ACTIONS,
    VITALS_SCALE,
    ActorCritic,
    KeyMemoryActorCritic,
    RecurrentActorCritic,
    encode_observation,
)
from training.task_rewards import TaskRewardTracker
from training.ppo import PPOConfig, compute_gae, ppo_update  # noqa: E402
from training.progression.tracker import ProgressionEventTracker, progression_step_reward  # noqa: E402
from training.recurrent_ppo import build_sequence_chunks, recurrent_ppo_update  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "curriculum" / "manifest.json"


def _rel_or_abs(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def dependency_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("doom-env", "vizdoom", "gymnasium", "numpy", "torch"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def shaped_reward(terminated: bool, info: dict, cfg: PPOConfig) -> float:
    """Exit/death reward plus small per-decision cost; no movement bonus."""
    r = cfg.step_cost
    if terminated:
        r += cfg.exit_reward if info.get("exit_reached", False) else cfg.death_penalty
    return float(r)


def save_checkpoint(
    path: Path,
    model: ActorCritic | RecurrentActorCritic | KeyMemoryActorCritic,
    cfg: PPOConfig,
    train_maps: list[dict],
    decisions_done: int,
    untrained: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": model.arch,
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "hidden_dim": model.hidden_dim,
        "num_actions": NUM_ACTIONS,
        "vitals_scale": VITALS_SCALE.tolist(),
        "contract_version": CONTRACT_VERSION,
        "seed": cfg.seed,
        "decisions_done": int(decisions_done),
        "total_decisions": cfg.total_decisions,
        "untrained": bool(untrained),
        "train_maps": [{"name": m["name"], "sha256": m["sha256"], "sampling_weight": m.get("sampling_weight", 1.0)} for m in train_maps],
    }
    if isinstance(model, (RecurrentActorCritic, KeyMemoryActorCritic)):
        payload["gru_hidden_dim"] = model.gru_hidden_dim
    if isinstance(model, KeyMemoryActorCritic):
        payload["num_slots"] = model.num_slots
        payload["slot_dim"] = model.slot_dim
    torch.save(payload, path)


def save_resumable_state(
    path: Path,
    model: ActorCritic | RecurrentActorCritic | KeyMemoryActorCritic,
    optimizer: torch.optim.Optimizer,
    cfg: PPOConfig,
    train_maps: list[dict],
    decisions_done: int,
    update_idx: int,
    episode_idx: int,
    action_gen: torch.Generator,
    np_rng: np.random.Generator,
    map_rng: np.random.Generator,
    completed_episodes: list[dict],
    update_logs: list[dict],
    wall_seconds: float = 0.0,
) -> None:
    """Save full resumable training state (model, optimizer, RNGs, config, progress)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": model.arch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "action_gen_state": action_gen.get_state(),
        "np_rng_state": np_rng.bit_generator.state,
        "map_rng_state": map_rng.bit_generator.state,
        "config": asdict(cfg),
        "input_dim": INPUT_DIM,
        "hidden_dim": model.hidden_dim,
        "num_actions": NUM_ACTIONS,
        "vitals_scale": VITALS_SCALE.tolist(),
        "contract_version": CONTRACT_VERSION,
        "decisions_done": int(decisions_done),
        "update_idx": int(update_idx),
        "episode_idx": int(episode_idx),
        "completed_episodes": completed_episodes,
        "update_logs": update_logs,
        "wall_seconds": float(wall_seconds),
        "train_maps": [{"name": m["name"], "sha256": m["sha256"], "sampling_weight": m.get("sampling_weight", 1.0)} for m in train_maps],
    }
    if isinstance(model, (RecurrentActorCritic, KeyMemoryActorCritic)):
        payload["gru_hidden_dim"] = model.gru_hidden_dim
    if isinstance(model, KeyMemoryActorCritic):
        payload["num_slots"] = model.num_slots
        payload["slot_dim"] = model.slot_dim
    torch.save(payload, path)


def ensure_fresh_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory {out_dir}. "
            "Pass --overwrite or choose a new --out-dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def train(
    cfg: PPOConfig,
    manifest_path: Path,
    out_dir: Path,
    save_weights: Path | None,
    overwrite: bool = False,
    checkpoint_every: int = 0,
    resume_from: Path | None = None,
) -> dict:
    if resume_from is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ensure_fresh_dir(out_dir, overwrite=overwrite)
    if checkpoint_every < 0 or (checkpoint_every and checkpoint_every % cfg.rollout_steps):
        raise ValueError("checkpoint_every must be zero or a positive multiple of rollout_steps")
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    action_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    map_rng = np.random.default_rng(cfg.seed + 2)

    manifest = json.loads(manifest_path.read_text())
    train_maps = [m for m in manifest["maps"] if m["split"] in ("train", "practice")]
    if not train_maps:
        raise ValueError(f"No training maps found in {manifest_path}")
    for m in train_maps:
        wad_path = ROOT / m["wad_path"]
        assert sha256_file(wad_path) == m["sha256"], f"Map SHA-256 mismatch for {wad_path}"

    map_probs = np.array([m.get("sampling_weight", 1.0) for m in train_maps], dtype=float)
    if not np.isfinite(map_probs).all() or (map_probs <= 0).any():
        raise ValueError("Map sampling weights must be finite and positive")
    map_probs /= map_probs.sum()

    model: ActorCritic | RecurrentActorCritic | KeyMemoryActorCritic
    if cfg.arch == "gru":
        model = RecurrentActorCritic()
    elif cfg.arch == "key_mem":
        model = KeyMemoryActorCritic()
    else:
        model = ActorCritic()
    is_recurrent = isinstance(model, (RecurrentActorCritic, KeyMemoryActorCritic))
    is_key_mem = isinstance(model, KeyMemoryActorCritic)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    initial_params = {k: v.clone() for k, v in model.state_dict().items()}

    num_updates = cfg.total_decisions // cfg.rollout_steps
    start_update = 1
    episode_idx = 0
    decisions_done = 0
    decisions_at_resume = 0
    prior_wall_seconds = 0.0
    completed_episodes: list[dict] = []
    update_logs: list[dict] = []

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        if str(ckpt.get("contract_version")) != CONTRACT_VERSION:
            raise ValueError(
                f"Resume contract_version mismatch: {ckpt.get('contract_version')!r} != {CONTRACT_VERSION!r}"
            )
        if str(ckpt.get("arch")) != cfg.arch:
            raise ValueError(f"Resume arch mismatch: {ckpt.get('arch')!r} != {cfg.arch!r}")
        if int(ckpt.get("input_dim", -1)) != INPUT_DIM or int(ckpt.get("num_actions", -1)) != NUM_ACTIONS:
            raise ValueError("Resume input_dim/num_actions mismatch")
        ckpt_scale = np.asarray(ckpt.get("vitals_scale", []), dtype=np.float32)
        if ckpt_scale.shape != VITALS_SCALE.shape or not np.allclose(ckpt_scale, VITALS_SCALE):
            raise ValueError("Resume vitals_scale mismatch")
        expected_maps = [{"name": m["name"], "sha256": m["sha256"], "sampling_weight": m.get("sampling_weight", 1.0)} for m in train_maps]
        saved_maps = [{**m, "sampling_weight": m.get("sampling_weight", 1.0)} for m in ckpt.get("train_maps", [])]
        if saved_maps != expected_maps:
            raise ValueError("Resume train_maps/hash mismatch")
        ckpt_cfg = ckpt.get("config", {})
        cur_cfg = asdict(cfg)
        for k in (
            "rollout_steps",
            "minibatch_size",
            "update_epochs",
            "lr",
            "gamma",
            "gae_lambda",
            "clip_coef",
            "vf_coef",
            "ent_coef",
            "max_grad_norm",
            "exit_reward",
            "death_penalty",
            "step_cost",
            "key_reward",
            "door_passable_reward",
            "kill_reward",
            "damage_penalty",
            "exploration_reward",
            "episode_ticks",
            "seed",
            "seq_len",
        ):
            if ckpt_cfg.get(k, 0.0) != cur_cfg.get(k):
                raise ValueError(f"Resume config mismatch on {k!r}: {ckpt_cfg.get(k)!r} != {cur_cfg.get(k)!r}")
        if cfg.total_decisions < int(ckpt["decisions_done"]):
            raise ValueError(
                f"Resume total_decisions ({cfg.total_decisions}) < saved decisions_done ({ckpt['decisions_done']})"
            )

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        action_gen.set_state(ckpt["action_gen_state"])
        np_rng.bit_generator.state = ckpt["np_rng_state"]
        map_rng.bit_generator.state = ckpt["map_rng_state"]
        decisions_done = int(ckpt["decisions_done"])
        decisions_at_resume = decisions_done
        prior_wall_seconds = float(ckpt.get("wall_seconds", 0.0))
        start_update = int(ckpt["update_idx"]) + 1
        episode_idx = int(ckpt["episode_idx"])
        completed_episodes = list(ckpt["completed_episodes"])
        update_logs = list(ckpt["update_logs"])

    envs = [
        make_env(ROOT / m["wad_path"], m["map_name"], episode_ticks=cfg.episode_ticks, render_buffers=False)
        for m in train_maps
    ]

    # On fresh start or resume, start a fresh episode (ViZDoom C++ state is not serialized).
    cur_map_idx = int(map_rng.choice(len(train_maps), p=map_probs))
    cur_env = envs[cur_map_idx]
    cur_seed = cfg.seed + episode_idx
    obs, _ = cur_env.reset(seed=cur_seed)
    obs_vec = encode_observation(obs)
    h_state = model.initial_state(1, "cpu") if is_recurrent else None

    def _new_tracker(map_meta: dict, env_obj) -> ProgressionEventTracker | None:
        if "key_color" not in map_meta:
            return None
        door_sec = int(np.where(env_obj.privileged_state()["door_lock"] > 0)[0][0])
        key_idx = VITALS_INDEX[f"key_{map_meta['key_color']}_card"]
        return ProgressionEventTracker(door_sec=door_sec, key_vital_idx=key_idx)

    cur_tracker = _new_tracker(train_maps[cur_map_idx], cur_env)
    task_tracker = TaskRewardTracker(cur_env.privileged_state())
    task_totals = dict(kill_bonus=0., damage_cost=0., exploration_bonus=0.)
    ep_decisions = 0
    ep_env_reward = 0.0
    ep_train_reward = 0.0
    ep_key_bonus = 0.0
    ep_door_bonus = 0.0
    ep_term_bonus = 0.0
    ep_step_cost = 0.0

    t_start = time.perf_counter()

    try:
        for update in range(start_update, num_updates + 1):
            b_obs = np.zeros((cfg.rollout_steps, INPUT_DIM), dtype=np.float32)
            b_actions = np.zeros((cfg.rollout_steps, NUM_ACTIONS), dtype=np.float32)
            b_log_probs = np.zeros(cfg.rollout_steps, dtype=np.float32)
            b_rewards = np.zeros(cfg.rollout_steps, dtype=np.float32)
            b_values = np.zeros(cfg.rollout_steps, dtype=np.float32)
            b_terminated = np.zeros(cfg.rollout_steps, dtype=bool)
            b_truncated = np.zeros(cfg.rollout_steps, dtype=bool)
            b_trunc_bootstrap = np.zeros(cfg.rollout_steps, dtype=np.float32)
            b_h_in = (
                np.zeros((cfg.rollout_steps, model.gru_hidden_dim), dtype=np.float32)
                if is_recurrent
                else None
            )
            b_mem_in = (
                np.zeros((cfg.rollout_steps, model.num_slots, model.slot_dim), dtype=np.float32)
                if is_key_mem
                else None
            )

            for step in range(cfg.rollout_steps):
                b_obs[step] = obs_vec
                obs_t = torch.from_numpy(obs_vec).unsqueeze(0)
                if is_recurrent:
                    assert h_state is not None and b_h_in is not None
                    if is_key_mem:
                        assert b_mem_in is not None
                        b_h_in[step] = h_state[0].squeeze(0).squeeze(0).detach().numpy()
                        b_mem_in[step] = h_state[1].squeeze(0).detach().numpy()
                    else:
                        b_h_in[step] = h_state.squeeze(0).squeeze(0).detach().numpy()
                    action_t, log_prob_t, value_t, next_h = model.sample_step(
                        obs_t, h_state, generator=action_gen
                    )
                else:
                    action_t, log_prob_t, value_t = model.sample_action(obs_t, generator=action_gen)
                    next_h = None

                if cur_tracker is not None:
                    cur_tracker.observe_pre_step(obs)
                action_np = action_t.squeeze(0).numpy().astype(np.int8)
                next_obs, env_r, term, trunc, info = cur_env.step(action_np)
                if cur_tracker is not None:
                    st = cur_env.privileged_state()
                    was_key = cur_tracker.key_picked_up
                    was_pass = cur_tracker.door_passable_after_key
                    cur_tracker.record_post_step(
                        next_obs,
                        st["sector_floor"],
                        st["sector_ceiling"],
                        bool(term and info.get("exit_reached", False)),
                    )
                    comps = progression_step_reward(
                        cur_tracker,
                        was_key,
                        was_pass,
                        bool(term),
                        bool(info.get("exit_reached", False)),
                        key_reward=cfg.key_reward,
                        door_passable_reward=cfg.door_passable_reward,
                        exit_reward=cfg.exit_reward,
                        death_penalty=cfg.death_penalty,
                        step_cost=cfg.step_cost,
                    )
                else:
                    term_b = (cfg.exit_reward if info.get("exit_reached", False) else cfg.death_penalty) if term else 0.0
                    comps = {
                        "step_cost": cfg.step_cost,
                        "key_bonus": 0.0,
                        "door_passable_bonus": 0.0,
                        "terminal_bonus": term_b,
                        "total": cfg.step_cost + term_b,
                    }
                task_comps = task_tracker.step(cur_env.privileged_state(), cfg.kill_reward, cfg.damage_penalty, cfg.exploration_reward)
                for key, value in task_comps.items():
                    task_totals[key] += value
                tr_r = comps["total"] + sum(task_comps.values())

                b_actions[step] = action_np.astype(np.float32)
                b_log_probs[step] = float(log_prob_t.item())
                b_values[step] = float(value_t.item())
                b_rewards[step] = tr_r
                b_terminated[step] = bool(term)
                b_truncated[step] = bool(trunc)
                # doom_env returns self._last_obs (stale final observation) on episode end.
                # On time-limit truncation, we bootstrap from V(obs_t, h_t) using the
                # pre-truncation observation and its matching recurrent state.
                if trunc and not term:
                    b_trunc_bootstrap[step] = float(value_t.item())

                ep_decisions += 1
                ep_env_reward += float(env_r)
                ep_train_reward += tr_r
                ep_key_bonus += comps["key_bonus"]
                ep_door_bonus += comps["door_passable_bonus"]
                ep_term_bonus += comps["terminal_bonus"]
                ep_step_cost += comps["step_cost"]
                decisions_done += 1

                if term or trunc:
                    outcome = "exit" if (term and info.get("exit_reached", False)) else ("death" if term else "timeout")
                    completed_episodes.append(
                        {
                            **task_totals,
                            "kills": task_tracker.kills,
                            "damage_taken": task_tracker.damage,
                            "unique_cells": len(task_tracker.visited),
                            "episode": episode_idx,
                            "layout": train_maps[cur_map_idx]["name"],
                            "env_seed": cur_seed,
                            "decisions": ep_decisions,
                            "nominal_action_ticks": ep_decisions * FRAME_SKIP,
                            "outcome": outcome,
                            "env_reward": ep_env_reward,
                            "train_reward": ep_train_reward,
                            "key_bonus": ep_key_bonus,
                            "door_passable_bonus": ep_door_bonus,
                            "terminal_bonus": ep_term_bonus,
                            "step_cost": ep_step_cost,
                            "key_picked_up": cur_tracker.key_picked_up if cur_tracker else False,
                            "door_passable_after_key": cur_tracker.door_passable_after_key if cur_tracker else False,
                        }
                    )
                    episode_idx += 1
                    cur_map_idx = int(map_rng.choice(len(train_maps), p=map_probs))
                    cur_env = envs[cur_map_idx]
                    cur_seed = cfg.seed + episode_idx
                    obs, _ = cur_env.reset(seed=cur_seed)
                    obs_vec = encode_observation(obs)
                    cur_tracker = _new_tracker(train_maps[cur_map_idx], cur_env)
                    if is_recurrent:
                        h_state = model.initial_state(1, "cpu")
                    task_tracker = TaskRewardTracker(cur_env.privileged_state())
                    task_totals = dict(kill_bonus=0., damage_cost=0., exploration_bonus=0.)
                    ep_decisions = 0
                    ep_env_reward = 0.0
                    ep_train_reward = 0.0
                    ep_key_bonus = 0.0
                    ep_door_bonus = 0.0
                    ep_term_bonus = 0.0
                    ep_step_cost = 0.0
                else:
                    obs = next_obs
                    obs_vec = encode_observation(obs)
                    if is_recurrent and next_h is not None:
                        if is_key_mem:
                            h_state = (next_h[0].detach(), next_h[1].detach())
                        else:
                            h_state = next_h.detach()

            with torch.no_grad():
                obs_next_t = torch.from_numpy(obs_vec).unsqueeze(0)
                if is_recurrent:
                    assert h_state is not None
                    _, next_val_t, _ = model.forward_step(obs_next_t, h_state)
                else:
                    _, next_val_t = model(obs_next_t)
                next_value = float(next_val_t.item())

            advantages, returns = compute_gae(
                rewards=b_rewards,
                values=b_values,
                terminated=b_terminated,
                truncated=b_truncated,
                trunc_bootstrap_values=b_trunc_bootstrap,
                next_value=next_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            if is_recurrent:
                assert b_h_in is not None
                seq_batch = build_sequence_chunks(
                    obs=b_obs,
                    actions=b_actions,
                    log_probs=b_log_probs,
                    advantages=advantages,
                    returns=returns,
                    h_in=b_h_in,
                    dones=b_terminated | b_truncated,
                    seq_len=cfg.seq_len,
                    mem_in=b_mem_in,
                )
                metrics = recurrent_ppo_update(
                    model=model,
                    optimizer=optimizer,
                    seq_batch=seq_batch,
                    cfg=cfg,
                    rng=np_rng,
                )
            else:
                metrics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    obs_batch=torch.from_numpy(b_obs),
                    actions_batch=torch.from_numpy(b_actions),
                    old_log_probs_batch=torch.from_numpy(b_log_probs),
                    advantages_batch=torch.from_numpy(advantages),
                    returns_batch=torch.from_numpy(returns),
                    cfg=cfg,
                    rng=np_rng,
                )
            for k, v in metrics.items():
                assert np.isfinite(v), f"Non-finite metric {k}={v} at update {update}"

            elapsed = time.perf_counter() - t_start
            session_decisions = decisions_done - decisions_at_resume
            recent_eps = completed_episodes[-20:]
            update_logs.append(
                {
                    "update": update,
                    "decisions": decisions_done,
                    "completed_episodes": len(completed_episodes),
                    "throughput_dps": session_decisions / elapsed if elapsed > 0 else 0.0,
                    "wall_s": prior_wall_seconds + elapsed,
                    "mean_env_reward_20": float(np.mean([e["env_reward"] for e in recent_eps])) if recent_eps else 0.0,
                    "mean_train_reward_20": float(np.mean([e["train_reward"] for e in recent_eps])) if recent_eps else 0.0,
                    "exit_rate_20": float(np.mean([1.0 if e["outcome"] == "exit" else 0.0 for e in recent_eps])) if recent_eps else 0.0,
                    **metrics,
                }
            )

            if checkpoint_every > 0 and decisions_done % checkpoint_every == 0:
                ckpt_dir = out_dir / "checkpoints"
                save_checkpoint(
                    ckpt_dir / f"ckpt_{decisions_done}.pt",
                    model,
                    cfg,
                    train_maps,
                    decisions_done=decisions_done,
                )
                save_resumable_state(
                    ckpt_dir / f"train_state_{decisions_done}.pt",
                    model,
                    optimizer,
                    cfg,
                    train_maps,
                    decisions_done=decisions_done,
                    update_idx=update,
                    episode_idx=episode_idx,
                    action_gen=action_gen,
                    np_rng=np_rng,
                    map_rng=map_rng,
                    completed_episodes=completed_episodes,
                    update_logs=update_logs,
                    wall_seconds=prior_wall_seconds + elapsed,
                )
    finally:
        for e in envs:
            e.close()

    session_wall = time.perf_counter() - t_start
    total_wall = prior_wall_seconds + session_wall
    session_decisions = decisions_done - decisions_at_resume
    param_l2_delta = float(
        sum((model.state_dict()[k] - initial_params[k]).pow(2).sum().item() for k in initial_params) ** 0.5
    )

    ckpt_path = out_dir / "checkpoint.pt"
    save_checkpoint(ckpt_path, model, cfg, train_maps, decisions_done=decisions_done)
    if save_weights is not None:
        save_checkpoint(save_weights, model, cfg, train_maps, decisions_done=decisions_done)

    with (out_dir / "updates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(update_logs[0].keys()))
        writer.writeheader()
        writer.writerows(update_logs)

    if completed_episodes:
        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in completed_episodes for k in row)))
            writer.writeheader()
            writer.writerows(completed_episodes)

    exits = sum(1 for e in completed_episodes if e["outcome"] == "exit")
    param_count = int(sum(p.numel() for p in model.parameters()))
    summary = {
        "contract_version": CONTRACT_VERSION,
        "command": " ".join(sys.argv),
        "git": git_revision(),
        "versions": dependency_versions(),
        "config": asdict(cfg),
        "param_count": param_count,
        "manifest_path": _rel_or_abs(manifest_path),
        "train_maps": [{"name": m["name"], "sha256": m["sha256"], "sampling_weight": m.get("sampling_weight", 1.0)} for m in train_maps],
        "truncation_bootstrap_note": (
            "doom_env returns self._last_obs on episode end; on time-limit truncation, "
            "GAE bootstraps from V(obs_t, h_t) using the pre-truncation observation and "
            "matching recurrent state rather than a post-step state or next episode reset."
        ),
        "stored_hidden_state_note": (
            "Recurrent PPO preserves each sequence's initial hidden state h0 from rollout "
            "collection and holds h0 fixed across minibatches/epochs within the PPO update."
        ),
        "decisions_done": decisions_done,
        "completed_episodes": len(completed_episodes),
        "completed_exits": exits,
        "completed_exit_rate": exits / len(completed_episodes) if completed_episodes else 0.0,
        "total_optimizer_steps": int(sum(u["optimizer_steps"] for u in update_logs)),
        "mean_sequence_count_per_update": float(np.mean([u["sequence_count"] for u in update_logs])),
        "mean_valid_transitions_per_minibatch": float(
            np.mean([u["mean_valid_transitions_per_minibatch"] for u in update_logs])
        ),
        "wall_seconds": total_wall,
        "session_wall_seconds": session_wall,
        "throughput_decisions_per_sec": session_decisions / session_wall if session_wall > 0 else 0.0,
        "param_l2_delta": param_l2_delta,
        "checkpoint_bytes": ckpt_path.stat().st_size,
        "final_update": update_logs[-1],
        "checkpoint_path": _rel_or_abs(ckpt_path),
        "saved_weights_path": _rel_or_abs(save_weights) if save_weights else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--arch", choices=("mlp", "gru", "key_mem"), default="mlp")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--total-decisions", type=int, default=51200)
    ap.add_argument("--rollout-steps", type=int, default=512)
    ap.add_argument("--minibatch-size", type=int, default=128)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--key-reward", type=float, default=0.0)
    ap.add_argument("--door-passable-reward", type=float, default=0.0)
    ap.add_argument("--kill-reward", type=float, default=0.0)
    ap.add_argument("--damage-penalty", type=float, default=0.0)
    ap.add_argument("--exploration-reward", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--save-weights", type=Path, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = PPOConfig(
        total_decisions=args.total_decisions,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        lr=args.lr,
        key_reward=args.key_reward,
        door_passable_reward=args.door_passable_reward,
        kill_reward=args.kill_reward,
        damage_penalty=args.damage_penalty,
        exploration_reward=args.exploration_reward,
        seed=args.seed,
        arch=args.arch,
        seq_len=args.seq_len,
    )
    summary = train(
        cfg,
        args.manifest,
        args.out_dir,
        args.save_weights,
        overwrite=args.overwrite,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
