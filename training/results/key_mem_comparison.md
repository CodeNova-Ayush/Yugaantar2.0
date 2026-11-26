# Key-Conditioned Memory (`key_mem`) vs. MLP & GRU Comparison (Stage 5B Budget)

All three architectures were trained from scratch on the 6 `training/progression/manifest.json` training layouts for `102,400` decisions across seeds `42, 43, 44` with identical PPO and reward settings (`key_reward=0.25`, `door_passable_reward=0.5`, `exit_reward=1.0`, `step_cost=-0.002`). Test layouts were not evaluated.

## Compute & Latency

| Architecture | Params | Checkpoint Size | Mean Wall Time (3 seeds) | Optimizer Steps / Seed | `act()` Latency p50 (p99) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp` | 77,450 | 314,615 B | 37.1s | 3,200 | 0.070 ms (0.143 ms) |
| `gru` | 160,010 | 645,365 B | 61.4s | 3,972–3,976 | 0.092 ms (0.161 ms) |
| `key_mem` | 181,678 | 733,749 B | 109.4s | 3,964–3,972 | 0.137 ms (0.233 ms) |

## Progression Train & Validation Results (20 Episodes / Layout)

| Architecture | Train Exits (360 eps) | Train Keys | Val Exits (120 eps) | Val Keys | Val Return-After-Key | Val Passable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp` | 181/360 (50.3%) | 229/360 (63.6%) | **22/120 (18.3%)** | 34/120 (28.3%) | 26/34 | 24/34 (70.6%) |
| `gru` | **188/360 (52.2%)** | 282/360 (78.3%) | 4/120 (3.3%) | 33/120 (27.5%) | 4/33 | 4/33 (12.1%) |
| `key_mem` | 158/360 (43.9%) | 284/360 (78.9%) | 0/120 (0.0%) | 1/120 (0.8%) | 1/1 | 1/1 (100.0%) |
| `key_mem` (memory-clear ablation, eval-only) | — | — | 1/120 (0.8%) | 3/120 (2.5%) | 2/3 | 1/3 (33.3%) |

## Replay & Telemetry Validation

- Inspected `prog_key_mem_eval_s42_val_replays/replays/prog_val_t_branch_e_ep0_seed1000.lmp` and `prog_val_u_detour_s_ep0_seed1000.lmp` via `playback_lmp`: both matched recorded `timeout` (`no_key`, `525` decisions).
- Staged validator (`prog_key_mem_s42/checkpoint.pt`, SHA-256 `692b77b519ff079c38d074c3b857fd689acec78842662b13578f00bab82c4d0c`): **PASSED with 3 warnings** (`validate_staging_output.txt`).

## Recommendation

Keep the **simpler MLP** as the reference architecture. Under this 6-map, `102,400`-decision regime, `key_mem` overfits to training layouts (`78.9%` train keys vs `0.8%` val keys, `0/120` val exits) at ~`2.9×` the training time of MLP.
