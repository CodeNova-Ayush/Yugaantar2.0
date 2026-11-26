"""One command for a bounded CPU campaign and validation-only selection."""
import argparse
from pathlib import Path
from training.ppo import PPOConfig
from training.train_ppo import train
from training.mixed.select import select
from training.mixed.generate import ROOT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--decisions',type=int,default=1024000);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    cfg=PPOConfig(total_decisions=a.decisions,seed=a.seed,key_reward=.25,door_passable_reward=.5,kill_reward=.3,damage_penalty=.002,exploration_reward=.01)
    train(cfg,ROOT/'training/mixed/manifest.json',a.out_dir,None,checkpoint_every=102400)
    print(select(a.out_dir))
