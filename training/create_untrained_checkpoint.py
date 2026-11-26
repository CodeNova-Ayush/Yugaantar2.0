"""Create a reproducible untrained ActorCritic checkpoint with recorded initialization seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.model import ActorCritic  # noqa: E402
from training.ppo import PPOConfig  # noqa: E402
from training.train_ppo import save_checkpoint  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "curriculum" / "manifest.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "curriculum_untrained" / "checkpoint.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint {args.out_path}")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = ActorCritic()
    cfg = PPOConfig(total_decisions=512, rollout_steps=512, seed=args.seed)

    manifest = json.loads(args.manifest.read_text())
    train_maps = [m for m in manifest["maps"] if m["split"] in ("train", "practice")]
    save_checkpoint(args.out_path, model, cfg, train_maps, decisions_done=0, untrained=True)

    meta = {
        "init_seed": args.seed,
        "untrained": True,
        "decisions_done": 0,
        "checkpoint_path": str(args.out_path.resolve().relative_to(ROOT)),
    }
    args.out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
