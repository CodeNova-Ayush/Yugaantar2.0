# Reproducing the mixed-task agent

Run commands from the repository root. Install the project's locked dependencies with
`uv sync --group dev`, or use the existing `.venv/bin/python` as below.
No network is needed by the submitted agent.

```sh
.venv/bin/python -m training.mixed.generate
.venv/bin/python -m training.mixed.verify
.venv/bin/python -m pytest tests/test_task_rewards.py tests/test_ppo.py tests/test_recurrent_ppo.py tests/test_key_memory.py tests/test_progression.py tests/test_env.py -q
.venv/bin/python training/train_ppo.py --manifest training/mixed/manifest.json --arch mlp --total-decisions 2048 --kill-reward 0.3 --damage-penalty 0.002 --exploration-reward 0.01 --key-reward 0.25 --door-passable-reward 0.5 --out-dir training/results/new_smoke
.venv/bin/python training/train_ppo.py --manifest training/mixed/manifest.json --arch mlp --total-decisions 1024000 --checkpoint-every 102400 --kill-reward 0.3 --damage-penalty 0.002 --exploration-reward 0.01 --key-reward 0.25 --door-passable-reward 0.5 --seed 42 --out-dir training/results/new_campaign
.venv/bin/python -m training.mixed.select training/results/new_campaign
```

The manifest samples **episodes**, not transitions, with probabilities 30% navigation,
40% key progression, 30% combat. Short episodes mean these proportions differ from
fractions of total training decisions. All task distributions and map hashes are logged.
There are 30 training layouts, 10 validation layouts and 10 test layouts. Combat
variants reuse navigation geometry ONLY within the same split. They are not 50
independent geometries. Geometry checks establish connectivity and enemy spawns;
they do not prove a policy can solve a map.

One decision lasts four engine ticks. PPO collects 512 decisions, estimates which
choices led to better subsequent outcomes, and updates the neural network for four
epochs. The actor outputs nine independent button probabilities. The critic predicts
future shaped reward and is used only during training. Opposing buttons are allowed
by the official contract; no scripted steering overrides the learned policy.

Rewards: exit +1, death -1, step -0.002, first key +0.25, first passable locked door
following a key +0.5, each kill +0.3, each damage point -0.002, first visit to a 64-unit
cell +0.01. Position, kill counters and door geometry are training/evaluation tools
ONLY. The agent never receives them. Reward components and raw environment reward
are logged separately, so shaped return must not be confused with the judge's score.

Checkpoint selection uses equal weighting of the three validation task-family exit
rates. Ties use combat kills per episode, fewer decisions, then earlier checkpoints.
Evaluation uses 20 episodes per map, environment seeds 0-19 and policy seeds 1000-1019.
This repeatedly used validation set is not an unbiased final test. Freeze weights
before evaluating test layouts. Existing navigation test layouts were used in earlier
experiments; disclose that historical exposure in results.

Resume requires the same algorithm, reward, seed and sampling settings. Set a larger
`--total-decisions` and pass `--resume path/to/train_state_N.pt`. Optimizer and random
states are restored, but the in-progress game is restarted; resumed trajectories are
not bit-for-bit identical to uninterrupted play. The official wrapper's terminal
observation is stale; time-limit value bootstrapping is an explicitly retained
approximation. Full training states are local artifacts; keep the final inference
weights and compact evidence in the submission.

For a separate machine use the same commands, seed and lockfile. The small model and
engine-heavy workload were tested on CPU with one Torch thread. No GPU speedup is
assumed.

For the complete bounded train-and-select workflow:

```sh
.venv/bin/python -m training.mixed.campaign --seed 43 --out-dir training/results/new_seed43
```

Never run two coding agents against these files simultaneously. The two completed
mixed campaigns use seeds 42 and 43. Historical experiments use other seeds and
budgets; do not combine their counts into the mixed campaign's sample size.
