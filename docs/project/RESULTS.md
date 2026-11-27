# Final measured results

Selected checkpoint: `training/results/mixed_1m_s43/checkpoints/ckpt_921600.pt`.

SHA-256: `46a80d55620011625b6c8c7090427ce191f6173fb60c5d8acb2949879fa27fff`.

| Task | Selected validation exits | Final test exits | Random test exits |
|---|---:|---:|---:|
| navigation | 80/80 | 80/80 | 31/80 |
| progression | 40/40 | 37/40 | 0/40 |
| combat | 79/80 | 78/80 | 7/80 |

Navigation test geometries were evaluated in earlier Stage 3 experiments. Combat variants share those geometries. Progression test policy evaluation was held back until this freeze. Not organizer hidden-map performance.

## Supplied map

{
  "episodes": 20,
  "exits": 7,
  "deaths": 0,
  "timeouts": 13,
  "kills_mean": 0.05,
  "door_passable_count": 9
}

## Training evidence

- mixed_1m_s42: 1,024,000 decisions; 8,594 completed episodes; 7,456 training exits; 465.5 seconds. Training exits are on-policy experience, not held-out evaluation.
- mixed_1m_s43: 1,024,000 decisions; 10,434 completed episodes; 9,568 training exits; 437.3 seconds. Training exits are on-policy experience, not held-out evaluation.

## Limits and verification

Two mixed training seeds were used. This is not a large-seed statistical study. Validation selection used ten checkpoints per seed plus the earlier progression reference. Test data was not used to pick the checkpoint. Combat maps contain one or two low-tier enemies and are substantially simpler than an unknown gauntlet.

See validator_output.txt for the actual integration run and verification.json for test counts. MacOS CPU pinning/RAM warnings and any missing Linux verification must remain disclosed. No hidden-map win rate is claimed.

## Curriculum v2 & `visible_entities` (`v2`) Investigation

To diagnose the `7/20` exit rate on `maps/two_rooms.wad`, we instrumented 20-episode rollouts and found that `ppo_pilot.pt` failed to open the unlocked door (`RAY_DOOR`) in `11/20` episodes (pressing `USE` while angled away or reversing the door 14–45 times) and averaged only `0.05` kills (`1/20`).

We added:
- `training/mixed_v2/`: D4-separated `unlocked_door` and multi-room `combat_rooms` layouts alongside `navigation` and `progression`.
- `obs_version="v2"` in `submission/model.py` (`INPUT_DIM_V2 = 482`, adding 17 compact `visible_entities` summary features while preserving `v1` 465-d checkpoint compatibility).
- `unlocked_door_reward` in `training/task_rewards.py`.

Comparing `ppo_pilot.pt` (`v1`, `mixed_1m_s43 ckpt_921600.pt`) vs the frozen `v2` candidate (`training/results/mixed_v2_1m_s43/checkpoints/ckpt_819200.pt`, SHA-256 `ede5d1664318a999e243b976396a68b23a431fcbed1c0183e1758226448a6e21`) on the reserved `mixed_v2` test split (20 episodes/layout) and `maps/two_rooms.wad`:

| Metric | `ppo_pilot.pt` (`v1`) | `mixed_v2_1m_s43` `ckpt_819200` (`v2`) |
|---|---:|---:|
| `mixed_v2` val balanced exit rate | 0.973 | 0.913 |
| `mixed_v2` test `navigation` | 80/80 (100%) | 80/80 (100%) |
| `mixed_v2` test `unlocked_door` | 60/60 (100%) | 45/60 (75.0%) |
| `mixed_v2` test `progression` | 37/40 (92.5%) | 4/40 (10.0%) |
| `mixed_v2` test `combat_rooms` | 69/80 (86.25%, 61 kills, 10 deaths) | 75/80 (93.75%, 107 kills, 1 death) |
| `mixed_v2` test balanced exit rate | 0.947 | 0.697 |
| `maps/two_rooms.wad` exits | 7/20 (35%) | 20/20 (100%) |
| `maps/two_rooms.wad` door passable / kills | 9/20 / 0.05 mean | 20/20 / 1.00 mean |

Because `ppo_pilot.pt` (`v1`) retains higher overall `mixed_v2` validation (`0.973` vs `0.913`) and test (`0.947` vs `0.697`) exit rates—particularly on key-and-door `progression` (`92.5%` vs `10.0%`)—`submission/weights/ppo_pilot.pt` is preserved as the default submission weights, while `mixed_v2_1m_s43/checkpoints/ckpt_819200.pt` is retained as the `v2` combat/two-rooms checkpoint.

## Reproducible Procedural Pipeline (`proc_v1`, 96 Maps)

We built `training/procedural/` (`proc_v1`: `64` train, `16` val, `16` test maps; D4-separated across all splits) and added a bounded LRU environment cache (`--env-cache-size 16`) in `training/train_ppo.py`. Running `proc_1m_s42` (`1,024,000` decisions, `3,983` episodes, `2,722` training exits, `1,005.1s`) selected `training/results/proc_1m_s42/checkpoints/ckpt_614400.pt` (SHA-256 `28f20b6bcef40f0c28c7fd080baeeb51eba88308a12e363e9a7111419b081e6c`) on validation (`0.9125` vs `0.8844` for `ppo_pilot.pt`).

| Metric | `ppo_pilot.pt` (`v1`) | `proc_1m_s42` `ckpt_614400` (`v2`) |
|---|---:|---:|
| `proc_v1` val balanced exit rate | 0.884 | **0.913** |
| `proc_v1` test `navigation` | **80/80 (100%)** | 72/80 (90.0%) |
| `proc_v1` test `unlocked_door` | **78/80 (97.5%)** | 72/80 (90.0%) |
| `proc_v1` test `progression` | **80/80 (100%)** | 49/80 (61.25%) |
| `proc_v1` test `combat` | **77/80 (96.25%, 48 kills, 1 death)** | 72/80 (90.0%, **97 kills, 0 deaths**) |
| `proc_v1` test balanced exit rate | **0.984** | 0.828 |
| `maps/two_rooms.wad` exits / kills | 7/20 (35%) / 0.05 mean | **16/20 (80%) / 0.95 mean** |

Because `ppo_pilot.pt` retains higher balanced `proc_v1` test performance (`0.984` vs `0.828`), `submission/weights/ppo_pilot.pt` is retained as the default submission weights.
