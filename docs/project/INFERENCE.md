# Inference and demo

The evaluator imports `DoomAgent` from `submission/agent.py`. It loads
`submission/weights/ppo_pilot.pt` once in its constructor. The filename is retained
for interface compatibility; FINAL_RESULTS.json records the actual selected model.
The final policy is a small PyTorch MLP. `reset()` resets the random generator, and
`act()` performs one forward pass and samples nine Bernoulli button choices.
No training data, manifest, network access, privileged game state or file writes
are needed at inference. PyTorch and NumPy are already in the official base image;
no additional submission dependencies are required.

```sh
# Contract/latency check (separate from gameplay measurement)
.venv/bin/python validate.py --device cpu
# Headless live agent demonstration
.venv/bin/python -m training.demo --record /tmp/doom-demo.lmp
# Native ViZDoom window; no OpenCV installation needed (pauses 1.5s at end by default)
.venv/bin/python -m training.demo --show --end-pause 1.5
# Repeatable 20-episode gameplay comparison; use a new output directory
.venv/bin/python training/eval_two_rooms.py --weights submission/weights/ppo_pilot.pt --out-dir /tmp/doom-final-eval
```

The agent samples actions. Different policy seeds can yield different outcomes.
Fixed environment and policy seeds make local comparisons reproducible, but CPU/GPU
floating-point differences can change trajectories. No CUDA equivalence claim is made.

The visible demo runs at game speed. Stop it with Ctrl+C in Terminal.
