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
