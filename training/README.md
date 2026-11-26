# Training

Put your training pipeline here: environment setup, reward shaping, curriculum, model code, logs.
Judges review this folder during manual review.

You may modify the engine, maps and game variables freely **for training**. Evaluation always uses
the vanilla engine through the official `doom_env` wrapper, so train on `doom_env` observations.

## Active Submission Checkpoint

- `submission/agent.py` currently loads `submission/weights/ppo_pilot.pt` (the Stage 2 `20,480`-decision checkpoint trained on `training/maps/practice_corridor.wad`).
- Stage 3 checkpoints (`training/results/curriculum_ppo_s42..s44/checkpoint.pt`) have **not** been promoted to `submission/weights/` based on test results.

## Legacy Stage 1–2 Experiments (Explicit Legacy Manifest)

Legacy maps live in `training/maps/manifest.json` (`practice_corridor` and `heldout_lshape`):

```bash
uv sync
uv run python training/generate_maps.py
uv run python training/verify_maps.py
uv run python training/eval_random.py --manifest training/maps/manifest.json --splits practice heldout --out-dir training/results/random_baseline_v2
uv run python training/train_ppo.py --manifest training/maps/manifest.json --total-decisions 2048 --seed 42 --out-dir training/results/ppo_smoke_2048
uv run python training/train_ppo.py --manifest training/maps/manifest.json --total-decisions 20480 --seed 42 --save-weights submission/weights/ppo_pilot.pt --out-dir training/results/ppo_pilot_20480
uv run python training/eval_policy.py --weights submission/weights/ppo_pilot.pt --manifest training/maps/manifest.json --splits practice heldout --out-dir training/results/ppo_pilot_20480/eval
```

## Stage 3: Varied 20-Map Navigation Curriculum

### Map Splits & Seeds

- **Curriculum manifest:** `training/curriculum/manifest.json` (`12` `train`, `4` `val`, `4` `test` layouts; D4-canonicalized ignoring `P` and `E` so no transformed geometry crosses splits).
- **Training seeds:** `42`, `43`, `44` (`51,200` decisions each, CPU 1 thread, `rollout_steps=512`, `minibatch_size=128`, `update_epochs=4`, `lr=0.0003`, `gamma=0.99`, `gae_lambda=0.95`, `clip_coef=0.2`).
- **Untrained baseline seed:** `init_seed=42` (`decisions_done=0`).
- **Evaluation seeds:** `env_seeds = [0..19]`, `policy_seeds = [1000..1019]`, `episode_ticks = 2100`. Evaluation defaults to `train`/`val` only; pass `--splits train val test` explicitly only after freezing configuration.

### Exact Stage 3 Commands

```bash
# 1. Generate & verify 20-map curriculum (checks D4 split hygiene and exit reachability)
uv run python training/curriculum/generate_curriculum.py
uv run python training/curriculum/verify_curriculum.py

# 2. Run focused regression tests
uv run python -m pytest tests/test_ppo.py tests/test_env.py -v

# 3. Smoke test (2,048 decisions) & 3x 51,200-decision pilots
uv run python training/train_ppo.py --manifest training/curriculum/manifest.json --total-decisions 2048 --seed 42 --out-dir training/results/curriculum_smoke_2048
uv run python training/train_ppo.py --manifest training/curriculum/manifest.json --total-decisions 51200 --seed 42 --out-dir training/results/curriculum_ppo_s42
uv run python training/train_ppo.py --manifest training/curriculum/manifest.json --total-decisions 51200 --seed 43 --out-dir training/results/curriculum_ppo_s43
uv run python training/train_ppo.py --manifest training/curriculum/manifest.json --total-decisions 51200 --seed 44 --out-dir training/results/curriculum_ppo_s44

# 4. Create reproducible untrained checkpoint (init_seed=42)
uv run python training/create_untrained_checkpoint.py --manifest training/curriculum/manifest.json --seed 42 --out-path training/results/curriculum_untrained/checkpoint.pt

# 5. Evaluate (defaults to train/val; pass 'test' explicitly only when frozen)
uv run python training/eval_random.py --manifest training/curriculum/manifest.json --splits train val test --out-dir training/results/curriculum_eval_random
uv run python training/eval_policy.py --weights training/results/curriculum_untrained/checkpoint.pt --manifest training/curriculum/manifest.json --splits train val test --out-dir training/results/curriculum_eval_untrained
uv run python training/eval_policy.py --weights training/results/curriculum_ppo_s42/checkpoint.pt --manifest training/curriculum/manifest.json --splits train val test --out-dir training/results/curriculum_eval_s42
uv run python training/eval_policy.py --weights training/results/curriculum_ppo_s43/checkpoint.pt --manifest training/curriculum/manifest.json --splits train val test --out-dir training/results/curriculum_eval_s43
uv run python training/eval_policy.py --weights training/results/curriculum_ppo_s44/checkpoint.pt --manifest training/curriculum/manifest.json --splits train val test --out-dir training/results/curriculum_eval_s44

# 6. Aggregate comparison from episode CSVs & demonstrate .lmp replay playback
uv run python training/summarize_stage3.py
uv run python training/replay_demo.py

# 7. Validate submission interface & dry-run
uv run python validate.py --map maps/two_rooms.wad:MAP01 --map training/curriculum/maps/val_pillar_bypass_e.wad:MAP01 --map training/curriculum/maps/test_chicane_e.wad:MAP01
```

### Stage 3 Results & Limitations

- **Results (`training/results/stage3_comparison.md`):**
  - `random`: Train `60.4%`, Val `62.5%`, Test `38.8%`
  - `untrained` (`seed=42`): Train `63.3%`, Val `66.2%`, Test `35.0%`
  - `ppo` (3-seed mean ± std): Train `80.4% ± 7.0%`, Val `85.4% ± 10.3%`, Test `69.6% ± 8.7%`
- **Limitations:**
  - **Not a controlled ablation:** Both training duration (`51,200` vs `20,480`) and map diversity (`12` vs `1`) changed from Stage 2.
  - **No exit sensor / no memory:** `doom_env` rays do not explicitly identify exits, and the feedforward MLP has no recurrent memory. Missing memory is one possible explanation (not yet an established cause) for timeouts on deeper dead-ends/spirals and seed-to-seed variance.

## Stage 4A: Recurrent PPO (GRU) Smoke Test & Proposed Comparison

### Smoke Test (`2,048` decisions, CPU, 1 thread)

```bash
uv run python -m pytest tests/test_ppo.py tests/test_recurrent_ppo.py -v
uv run python training/train_ppo.py --manifest training/curriculum/manifest.json --arch gru --seq-len 64 --total-decisions 2048 --seed 42 --out-dir training/results/gru_smoke_2048
```

### Equal-Budget 3-Seed MLP-vs-GRU Comparison (Stage 4B, Train/Val Only)

- **Equal game experience, not equal compute:** Both architectures ran `51,200` decisions on the 12 `train` maps (seeds `42, 43, 44`, identical PPO/reward settings). GRU uses `160,010` parameters (~`2.07x` MLP's `77,450`), ~`30.6s` wall time (vs ~`17.7s`), and `2,020–2,064` optimizer steps (vs `1,600`), with `~9.6` sequences/update and `~100–102` valid transitions/minibatch.
- **Stored-hidden-state approximation:** Recurrent PPO preserves each sequence's rollout-time `h0` and holds it fixed across minibatches/epochs within a PPO update.
- **Train & Val Results (stopped before `test`):**
  - `MLP`: Train `80.4% ± 7.0%` (`214.0` dec), Val `85.4% ± 10.3%` (`185.7` dec)
  - `GRU`: Train `77.2% ± 11.5%` (`213.0` dec), Val `88.3% ± 10.9%` (`169.8` dec)
- **Conclusion:** Validation evidence is **mixed** (slight val mean/speed edge for GRU, but overlapping seed variance and lower train mean). The simpler MLP remains the reference, and `submission/weights/ppo_pilot.pt` is unchanged.

## Stage 5A–5B: Key-and-Door Progression Benchmark & Baseline Training

- **Manifest:** `training/progression/manifest.json` (`6` `train`, `2` `val`, `2` `test` layouts; D4-separated by walkable geometry).
- **Mechanics & visibility measurements:** Vanilla Doom keycards (`Red`/`Blue`/`Yellow`) + UDMF `Door_LockedRaise` (`special=13`, `arg3=129/130/131`) + walk-over `Exit_Normal` (`special=243`). Measured in `training/results/progression_verify_v2/summary.json`: all 10 maps have key-to-door distances `166.9–459.4` units (`<= 1024.0` `RAY_RANGE`). `3` train maps (`prog_train_inline_red_e`, `prog_train_hook_yellow_n`, `prog_train_wide_hall_red_s`) have the door sensed at key pickup; in the other `7`, the door is not sensed in `rays` at key pickup (`1–38` steps absent beforehand, within `1,024` units).
- **Commands:**
  ```bash
  uv run python -m pytest tests/test_progression.py tests/test_ppo.py tests/test_recurrent_ppo.py -v
  uv run python training/progression/generate_progression.py
  uv run python training/progression/verify_progression.py --out-dir training/results/progression_verify_v2 --overwrite
  uv run python training/progression/eval_progression.py --splits train val --out-dir training/results/progression_eval_random_val_v2 --save-replays-dir training/results/progression_eval_random_val_v2/replays --overwrite

  # Stage 5B: 2,048-decision smoke tests & 102,400-decision pilots (seeds 42, 43, 44)
  uv run python training/train_ppo.py --manifest training/progression/manifest.json --arch mlp --key-reward 0.25 --door-passable-reward 0.5 --total-decisions 102400 --seed 42 --out-dir training/results/prog_mlp_s42
  uv run python training/train_ppo.py --manifest training/progression/manifest.json --arch gru --seq-len 64 --key-reward 0.25 --door-passable-reward 0.5 --total-decisions 102400 --seed 42 --out-dir training/results/prog_gru_s42
  uv run python training/progression/eval_progression.py --weights training/results/prog_mlp_s42/checkpoint.pt --splits train val --out-dir training/results/prog_mlp_eval_s42
  uv run python training/eval_policy.py --weights training/results/prog_mlp_s42/checkpoint.pt --manifest training/curriculum/manifest.json --splits val --out-dir training/results/prog_mlp_nav_val_s42
  ```
- **Stage 5B Results (`102,400` decisions from scratch, seeds `42, 43, 44`, `train`/`val` only):**
  - `MLP`: Prog Train `50.3%` (`181/360` exits), Prog Val `18.3%` (`22/120` exits; `34/120` keys; `24/34` passable), Nav Val `89.2% ± 9.8%`
  - `GRU`: Prog Train `52.2%` (`188/360` exits), Prog Val `3.3%` (`4/120` exits; `33/120` keys; `4/33` passable), Nav Val `90.0% ± 9.7%`
  - Main failure points on unseen progression `val` layouts: low key-collection on unseen branch geometries (`~28%`) and, especially for GRU (`4/33`), failing to return to and unlock the occluded door after collecting the key.
- **Next (not run yet):** Equal-experience comparison adding key-conditioned memory on `training/progression/manifest.json`.

## Key-Conditioned Memory (`key_mem`) Results & Limitations

- **Results (`102,400` decisions, seeds `42, 43, 44`, `training/results/key_mem_comparison.md`):**
  - `key_mem` (`181,678` params, `~109.4s`/seed): Prog Train `43.9%` (`158/360` exits, `284/360` keys), Prog Val **`0.0%`** (`0/120` exits, `1/120` keys). Memory-clear ablation (`eval`-only): `1/120` (`0.8%`) Val exits.
- **Limitations:** Overfits heavily to the 6 training layouts at ~`2.9x` MLP wall time. Keeping the simpler MLP as reference.

## Longer MLP Campaign (`1,024,000` Decisions, Seed `42`)

- **Resumable checkpoints:** `--checkpoint-every 102400` saves inference checkpoints (`checkpoints/ckpt_<decisions>.pt`) and full training states (`checkpoints/train_state_<decisions>.pt` with model, optimizer, Torch/NumPy/map RNG states, config, and `decisions_done`). Resuming via `--resume` restarts a fresh episode because ViZDoom's C++ engine state is not serialized.
- **Predeclared validation selection rule:** Select the checkpoint with highest progression `val` `unconditional_exit_rate`, breaking ties by highest `val` `key_pickup_rate`, then lowest `val` `decisions_mean`, then earliest `decisions_done`. Test split is never inspected.
- **1M Campaign & `two_rooms.wad` Commands:**
  ```bash
  uv run python training/train_ppo.py --manifest training/progression/manifest.json --arch mlp --key-reward 0.25 --door-passable-reward 0.5 --total-decisions 1024000 --checkpoint-every 102400 --seed 42 --out-dir training/results/prog_mlp_1m_s42
  uv run python training/eval_two_rooms.py --out-dir training/results/two_rooms_eval_random
  uv run python training/eval_two_rooms.py --weights training/results/prog_mlp_1m_s42/checkpoints/ckpt_1024000.pt --out-dir training/results/two_rooms_eval_mlp_1m
  ```
- **Results (`training/results/prog_mlp_1m_s42/learning_curve_report.md`):** `ckpt_1024000.pt` reached `26/40 (65.0%)` on `prog_val` (`26/26` return/passable/exit given key; `14/40` `no_key`) and `80/80 (100%)` on `nav_val`. On `maps/two_rooms.wad`, it opened the unlocked door in `13/20` episodes (`1/20` exit, `0/20` kills, `26.25` mean damage) vs `0/20` exits for random.

## Proposed Mixed Curriculum (Not Launched Yet)

- **Separate training layouts (preserving `val`/`test` separation):**
  1. **Navigation (~30%):** 12 varied corridor/turn/pillar layouts (including unlocked `RAY_DOOR` doors).
  2. **Key & Door Progression (~40%):** Expanded 12–16 key-and-door layouts covering more branch geometries and orientations to reduce `no_key`.
  3. **Combat & Survival (~30%):** Separate combat rooms with low-tier enemies, ammo/health pickups, and exits (using `visible_entities` + `rays`, adding `+0.2` kill bonus and `-0.01/hp` damage penalty).
- **Bounded budget & selection rule:** `1,024,000` decisions (`3` seeds `42, 43, 44`, PPO + MLP, checkpoint every `102,400`), selecting checkpoints by highest mean `val` exit rate across `nav_val`, `prog_val`, and `combat_val` (tie-breakers: `prog_val` passable rate → `combat_val` kill/survival rate → lowest mean decisions → earliest checkpoint). Default `submission/weights/ppo_pilot.pt` remains unchanged.

## Final mixed-task implementation

The submission-ready workflow and limitations are documented in
[`docs/project/REPRODUCE.md`](../docs/project/REPRODUCE.md), with final measured
results in [`docs/project/RESULTS.md`](../docs/project/RESULTS.md).
The mixed curriculum, event rewards, evaluation, validation-only selection and
freeze-before-test scripts are under `training/mixed/`. Inference and demo commands
are in [`docs/project/INFERENCE.md`](../docs/project/INFERENCE.md).
Earlier sections above are historical experiments, not the final selected model.
