# Expanded procedural experiment

Run: `.venv/bin/python -m training.expanded.campaign`

This is a bounded fresh PPO run with 200 training maps and 24 validation maps,
all checked for geometry duplication against the prior procedural/mixed manifests.
Existing weights are copied into the experiment before baseline evaluation.
The policy uses v2 official observations. Rewards and PPO settings remain fixed.
A 2,048-decision smoke test precedes 2,048,000 training decisions.
Twenty checkpoints are evaluated on 20 episodes per validation layout.

Replacement eligibility is fixed in protocol.json: improve balanced validation
exit rate with no task falling more than 5 percentage points below the default.
The script never replaces submission weights or uses test results.
Recommendation is saved in training/results/expanded_2m_s43/recommendation.json.

Map checks establish engine loading and generator connectivity, not exhaustive
physical door/exit traversal. Additional mechanical verification remains necessary.
Do not run a second campaign into the same directory or change files during training.
