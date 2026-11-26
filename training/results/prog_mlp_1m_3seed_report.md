# 1,024,000-Decision MLP Progression Campaign: 3-Seed Report (`seeds 42, 43, 44`)

All three runs trained from scratch on the 6 `training/progression/manifest.json` training layouts with identical PPO and reward settings (`arch=mlp`, `77,450` params, `key_reward=0.25`, `door_passable_reward=0.5`, `exit_reward=1.0`, `step_cost=-0.002`, `checkpoint_every=102400`). Default submission weights (`submission/weights/ppo_pilot.pt`) remain unchanged, and progression `test` layouts were not evaluated.

## Training Summary

| Seed | Decisions | Completed Eps (Exits) | Wall Time | Throughput | Optimizer Steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1,024,000 | 4,721 (3,557, 75.3%) | 387.03s | 2,646 dec/s | 32,000 |
| 43 | 1,024,000 | 6,831 (6,082, 89.0%) | 410.44s | 2,495 dec/s | 32,000 |
| 44 | 1,024,000 | 6,400 (5,703, 89.1%) | 402.71s | 2,543 dec/s | 32,000 |

## Validation Learning Curves (`prog_val`: 40 eps, `nav_val`: 80 eps)

| Decisions | `s42` ProgVal (Key/Pass) | `s42` NavVal | `s43` ProgVal (Key/Pass) | `s43` NavVal | `s44` ProgVal (Key/Pass) | `s44` NavVal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 102,400 | 17/40 (22/19) | 61/80 | 4/40 (11/4) | 73/80 | 1/40 (1/1) | 80/80 |
| 204,800 | 15/40 (18/16) | 62/80 | 8/40 (12/8) | 80/80 | 18/40 (18/18) | 80/80 |
| 307,200 | 10/40 (10/10) | 80/80 | 1/40 (1/1) | 79/80 | 30/40 (31/31) | 80/80 |
| 409,600 | 9/40 (9/9) | 80/80 | 0/40 (3/0) | 64/80 | 26/40 (39/39) | 80/80 |
| 512,000 | 3/40 (4/3) | 80/80 | 0/40 (0/0) | 80/80 | 18/40 (20/18) | 80/80 |
| 614,400 | 21/40 (28/21) | 80/80 | 0/40 (0/0) | 80/80 | 36/40 (37/36) | 80/80 |
| 716,800 | 0/40 (20/0) | 60/80 | 1/40 (1/1) | 80/80 | 20/40 (22/21) | 80/80 |
| 819,200 | 9/40 (10/9) | 80/80 | 24/40 (24/24) | 80/80 | 25/40 (28/25) | 80/80 |
| 921,600 | 1/40 (1/1) | 80/80 | **38/40 (38/38)** | 80/80 | 37/40 (37/37) | 80/80 |
| 1,024,000 | **26/40 (26/26)** | 80/80 | 27/40 (28/28) | 80/80 | **37/40 (37/37)** | 80/80 |

## Selected Checkpoints & `maps/two_rooms.wad` Evaluation (20 Seeds)

- **`s42`:** `training/results/prog_mlp_1m_s42/checkpoints/ckpt_1024000.pt` (`sha256: 10dbba3796666152f0c1d2a3afe9108409fa3c4cd49063ea784b345279553059`)
- **`s43`:** `training/results/prog_mlp_1m_s43/checkpoints/ckpt_921600.pt` (`sha256: fdb675414312c7f0111f5d25f731e58f47c6f001e94cd4200bbec0e5a078831f`)
- **`s44`:** `training/results/prog_mlp_1m_s44/checkpoints/ckpt_1024000.pt` (`sha256: e0b6f163a439176b1b4d99e75105032ab51cf7c46ff60a9eb38e053e2692a1cb`)
- **Selected `prog_val` 3-seed mean:** `101/120 (84.2% ± 13.6%)` exits (`101/101` return/passable/exit when key collected).

| Policy | Exits | Deaths | Timeouts | Door Init / Passable | Kills (Mean) | Mean Damage | Mean Decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SeededRandomPolicy` | 0/20 (0%) | 1/20 | 19/20 | 3/20 / 2/20 | 0.0 (0/20) | 7.65 | 524.65 |
| `s42` (`ckpt_1024000.pt`) | 1/20 (5%) | 0/20 | 19/20 | 13/20 / 13/20 | 0.0 (0/20) | 26.25 | 514.75 |
| `s43` (`ckpt_921600.pt`) | 9/20 (45%) | 0/20 | 11/20 | 18/20 / 17/20 | 0.0 (0/20) | 23.70 | 442.30 |
| `s44` (`ckpt_1024000.pt`) | 0/20 (0%) | 3/20 | 17/20 | 6/20 / 6/20 | 0.0 (0/20) | 21.60 | 513.80 |
| **3-Seed Total / Mean** | **10/60 (16.7%)** | **3/60** | **47/60** | **37/60 / 36/60** | **0.0 (0/60)** | **23.85** | **490.28** |
