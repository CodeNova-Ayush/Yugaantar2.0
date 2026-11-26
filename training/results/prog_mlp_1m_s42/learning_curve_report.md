# 1,024,000-Decision MLP Progression Campaign (`seed=42`)

Trained from scratch on the 6 `training/progression/manifest.json` training layouts with Stage 5B rewards (`key_reward=0.25`, `door_passable_reward=0.5`, `exit_reward=1.0`, `step_cost=-0.002`). Total runtime: `387.03s` (`2,646` dec/s, `4,721` completed episodes, `3,557` exits = `75.3%`, `32,000` optimizer steps).

## Learning Curve Across 10 Checkpoints (`val` Splits Only)

| Decisions | Prog Val Exits | Key Pickup | Return-After-Key | Passable Door | Ordered Seq | Prog Val Mean Dec | Nav Val Exits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 102,400 | 17/40 (42.5%) | 22/40 | 20/22 | 19/22 (86.4%) | 17/40 | 434.5 | 61/80 (76.2%) |
| 204,800 | 15/40 (37.5%) | 18/40 | 17/18 | 16/18 (88.9%) | 15/40 | 445.1 | 62/80 (77.5%) |
| 307,200 | 10/40 (25.0%) | 10/40 | 10/10 | 10/10 (100.0%) | 10/40 | 443.3 | 80/80 (100.0%) |
| 409,600 | 9/40 (22.5%) | 9/40 | 9/9 | 9/9 (100.0%) | 9/40 | 429.8 | 80/80 (100.0%) |
| 512,000 | 3/40 (7.5%) | 4/40 | 3/4 | 3/4 (75.0%) | 3/40 | 512.9 | 80/80 (100.0%) |
| 614,400 | 21/40 (52.5%) | 28/40 | 22/28 | 21/28 (75.0%) | 21/40 | 413.0 | 80/80 (100.0%) |
| 716,800 | 0/40 (0.0%) | 20/40 | 0/20 | 0/20 (0.0%) | 0/40 | 525.0 | 60/80 (75.0%) |
| 819,200 | 9/40 (22.5%) | 10/40 | 10/10 | 9/10 (90.0%) | 9/40 | 434.4 | 80/80 (100.0%) |
| 921,600 | 1/40 (2.5%) | 1/40 | 1/1 | 1/1 (100.0%) | 1/40 | 515.5 | 80/80 (100.0%) |
| **1,024,000** | **26/40 (65.0%)** | **26/40** | **26/26** | **26/26 (100.0%)** | **26/40** | **289.7** | **80/80 (100.0%)** |

## Selected Checkpoint & Replay / Staging Validation

- **Selected by predeclared `val` rule:** `checkpoints/ckpt_1024000.pt` (SHA-256 `10dbba3796666152f0c1d2a3afe9108409fa3c4cd49063ea784b345279553059`), achieving `11/20 (55%)` on `prog_val_t_branch_e` and `15/20 (75%)` on `prog_val_u_detour_s`, with `26/26` return/passable/exit once the key is collected and `80/80 (100%)` on `nav_val`. All 14 failures were `no_key`.
- **Replays (`selected_val_replays/replays/`):** Verified 4 `.lmp` files via `playback_lmp`: `t_branch_ep0` (`exit, 138`), `t_branch_ep3` (`timeout/no_key, 525`), `u_detour_ep0` (`exit, 236`), `u_detour_ep1` (`timeout/no_key, 525`).
- **Staged validator (`validate_staging_output.txt`):** **PASSED with 3 warnings** (temp dir not git, macOS CPU/RAM warnings), init `0.6s`, `p50 0.3 ms/step`.

## Evaluation on `maps/two_rooms.wad` (20 Seeds)

| Policy | Exits | Deaths | Timeouts | Door Initiated | Door Passable | Kills (Mean / Nonzero Eps) | Mean Damage Taken | Mean Decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SeededRandomPolicy` | 0/20 (0%) | 1/20 | 19/20 | 3/20 | 2/20 | 0.0 (0/20) | 7.65 | 524.65 |
| `ckpt_1024000.pt` (1M MLP) | 1/20 (5%) | 0/20 | 19/20 | 13/20 | 13/20 | 0.0 (0/20) | 26.25 | 514.75 |

- **Failure inspection:**
  - On `two_rooms.wad`, `ckpt_1024000.pt` opened the unlocked door (`RAY_DOOR`) in `13/20` episodes, but failed to open it in `7/20` (like `ep0`, where `door_seen_steps=473` and `door_ceil=[128, 128, 0, 128]`), likely because it only trained on `RAY_LOCKED` doors. Across all `20` episodes it scored `0` kills while taking `26.25` mean damage (never trained on combat).
  - On `prog_val`, all `14/40` failures were `no_key` (exploration on unseen branches), while all `26/26` key-collecting episodes returned, opened the door, and exited.
