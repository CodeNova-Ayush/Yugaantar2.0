# Desk review: explain and demonstrate

**What did you build?** A neural policy and custom PPO training/evaluation pipeline.
ViZDoom runs DOOM; the organizers' doom_env converts the game into official sensor
observations and applies nine-button actions. We did not build or replace that wrapper.

**What does the agent see?** 64 normalized ray distances, six hit categories encoded
as one-hot vectors, and 17 scaled vitals: 465 input values. It does not use RGB,
absolute position, map names, or the training reward counters at inference.

**What is PPO?** We let the policy play, estimate which choices were better than
expected using a value network, then change action probabilities. PPO clips its
policy objective to discourage excessively large changes on the same batch. This
is not a guarantee against performance regressions, which our checkpoint curves show.

**Why a simple model?** The MLP has 77,450 parameters. We implemented GRU and
key-conditioned external-memory alternatives, but their earlier progression
validation results did not justify using them. The MLP has no explicit episode
memory: successful backtracking does not prove that it remembered a door.

**Why mixed training?** Navigation-only training did not teach combat. Task sampling
exposes one policy to navigation, key progression and combat while keeping evaluation
layouts separate. This is a curriculum by task mixture, not an automatic difficulty
schedule.

**Is the reward the judge's score?** No. The judge's weights are hidden. Our shaping
provides learning signals, and we report exits, keys, doors, kills and deaths separately.

**Does a validator pass mean it wins?** No. It checks contract compliance, actions,
latency and resources. Gameplay is measured with repeated episodes on held-out layouts.

**Is this zero-shot?** The weights do not update during evaluation. New geometry can
still cause failure. Repeated validation selection and historical navigation test
exposure limit what can be claimed about generalization. No hidden organizer maps
were available to us.

**What is the USP?** A small, inspectable sensor-only PPO system with explicit event
telemetry and controlled memory experiments. Present measured strengths and failures;
do not claim novel memory improves performance when the evidence says otherwise.

Demo: show the final results JSON and a selected success AND failure replay. Explain
observations -> policy -> nine buttons -> four game ticks -> new observations. Keep the
validator output separate from gameplay evidence. All exact commands and checkpoint
hashes belong in the final results report.
