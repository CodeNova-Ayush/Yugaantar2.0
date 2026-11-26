# Stage 3 Aggregate Comparison (from Existing Episode CSVs)

## Split Summary

| Policy | Split | Exits / Eps | Exit Rate | Timeouts | Mean Decisions | Success-Only Mean Decisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random | train | 145/240 | 60.4% | 95 | 357.9 | 248.4 |
| random | val | 50/80 | 62.5% | 30 | 377.3 | 288.7 |
| random | test | 31/80 | 38.8% | 49 | 423.3 | 262.6 |
| untrained | train | 152/240 | 63.3% | 88 | 359.4 | 263.6 |
| untrained | val | 53/80 | 66.2% | 27 | 369.5 | 290.3 |
| untrained | test | 28/80 | 35.0% | 52 | 439.2 | 279.9 |
| ppo_s42 | train | 196/240 | 81.7% | 44 | 223.7 | 156.0 |
| ppo_s42 | val | 63/80 | 78.8% | 17 | 232.4 | 153.4 |
| ppo_s42 | test | 47/80 | 58.8% | 33 | 314.8 | 167.1 |
| ppo_s43 | train | 171/240 | 71.2% | 69 | 280.7 | 182.1 |
| ppo_s43 | val | 62/80 | 77.5% | 18 | 250.3 | 170.5 |
| ppo_s43 | test | 56/80 | 70.0% | 24 | 319.6 | 231.6 |
| ppo_s44 | train | 212/240 | 88.3% | 28 | 137.6 | 86.4 |
| ppo_s44 | val | 80/80 | 100.0% | 0 | 74.4 | 74.4 |
| ppo_s44 | test | 64/80 | 80.0% | 16 | 233.2 | 160.2 |

## PPO 3-Seed Variation (Seeds 42, 43, 44)

| Split | Exit Rate Mean ± Std (Min–Max) | Mean Decisions (± Std) | Success-Only Decisions (± Std) |
| --- | --- | --- | --- |
| train | 80.4% ± 7.0% (71.2%–88.3%) | 214.0 ± 58.8 | 141.5 ± 40.4 |
| val | 85.4% ± 10.3% (77.5%–100.0%) | 185.7 ± 79.0 | 132.8 ± 41.9 |
| test | 69.6% ± 8.7% (58.8%–80.0%) | 289.2 ± 39.6 | 186.3 ± 32.1 |

## Per-Layout Exit Rates & Success-Only Mean Decisions

| Layout | Split | Random | Untrained | PPO s42 | PPO s43 | PPO s44 |
| --- | --- | --- | --- | --- | --- | --- |
| train_straight_short_e | train | 19/20 (195) | 18/20 (195) | 20/20 (69) | 20/20 (103) | 20/20 (19) |
| train_straight_long_w | train | 8/20 (323) | 13/20 (280) | 20/20 (188) | 15/20 (242) | 20/20 (36) |
| train_wide_hall_n | train | 20/20 (198) | 20/20 (202) | 20/20 (72) | 20/20 (78) | 20/20 (20) |
| train_bend_left_90_e | train | 18/20 (189) | 18/20 (186) | 20/20 (69) | 19/20 (197) | 20/20 (97) |
| train_bend_right_90_s | train | 17/20 (232) | 13/20 (263) | 13/20 (309) | 20/20 (104) | 20/20 (49) |
| train_bend_wide_left_n | train | 12/20 (236) | 13/20 (252) | 20/20 (102) | 11/20 (241) | 19/20 (125) |
| train_s_curve_e | train | 8/20 (304) | 7/20 (333) | 11/20 (218) | 15/20 (245) | 20/20 (81) |
| train_s_curve_long_s | train | 8/20 (271) | 12/20 (370) | 11/20 (214) | 13/20 (227) | 20/20 (119) |
| train_u_turn_e | train | 12/20 (348) | 10/20 (346) | 20/20 (126) | 1/20 (218) | 0/20 (-) |
| train_t_junction_n | train | 12/20 (234) | 14/20 (293) | 18/20 (183) | 17/20 (108) | 18/20 (122) |
| train_staggered_rooms_e | train | 6/20 (356) | 6/20 (338) | 13/20 (249) | 9/20 (348) | 16/20 (194) |
| train_stepped_stair_n | train | 5/20 (379) | 8/20 (308) | 10/20 (277) | 11/20 (328) | 19/20 (118) |
| val_narrow_long_n | val | 8/20 (297) | 12/20 (286) | 17/20 (153) | 16/20 (226) | 20/20 (34) |
| val_l_asymmetric_w | val | 16/20 (280) | 15/20 (247) | 20/20 (105) | 8/20 (203) | 20/20 (132) |
| val_pillar_bypass_e | val | 11/20 (282) | 10/20 (292) | 18/20 (177) | 18/20 (183) | 20/20 (69) |
| val_hook_chamber_s | val | 15/20 (299) | 16/20 (333) | 8/20 (221) | 20/20 (103) | 20/20 (62) |
| test_chicane_e | test | 1/20 (399) | 1/20 (428) | 1/20 (359) | 2/20 (320) | 18/20 (218) |
| test_l_narrow_to_wide_s | test | 11/20 (263) | 10/20 (312) | 20/20 (145) | 18/20 (214) | 16/20 (159) |
| test_fork_deadend_w | test | 7/20 (238) | 5/20 (327) | 6/20 (255) | 17/20 (200) | 20/20 (71) |
| test_spiral_inward_n | test | 12/20 (265) | 12/20 (221) | 20/20 (153) | 19/20 (267) | 10/20 (236) |
