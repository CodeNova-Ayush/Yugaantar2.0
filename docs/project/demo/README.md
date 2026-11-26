# Representative official-map replays

These are actual final-policy episodes, not scripted solutions:

- `two_rooms_ep0_seed1000.lmp`: exit, 173 decisions (environment seed 0).
- `two_rooms_ep1_seed1001.lmp`: timeout, 525 decisions (environment seed 1).

Both were replayed in ViZDoom and matched the recorded outcomes. The map is
`maps/two_rooms.wad`, MAP01. The full 20-episode result is 7 exits, 13 timeouts,
zero deaths and one total kill; do not present the successful replay as typical.

Live demonstration of the successful fixed seed:

```sh
.venv/bin/python -m training.demo --seed 0 --policy-seed 1000 --show
```

The viewer uses ViZDoom's native window; OpenCV is not required.
Stop the demo with Ctrl+C in Terminal. Omit `--show` for headless execution.
