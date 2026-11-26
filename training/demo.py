"""Run the submitted agent on a map; optionally display RGB and record a replay.
Usage: python -m training.demo --show --record /tmp/demo.lmp
"""
import argparse
from pathlib import Path
from doom_env import make_env
from submission.agent import DoomAgent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',type=Path,default=Path('maps/two_rooms.wad'))
    p.add_argument('--map-name',default='MAP01');p.add_argument('--weights',type=Path)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--policy-seed',type=int,default=1000)
    p.add_argument('--show',action='store_true');p.add_argument('--record',type=Path)
    a=p.parse_args()
    if a.record and a.record.exists():raise FileExistsError(a.record)
    if a.record:a.record.parent.mkdir(parents=True,exist_ok=True)
    env=make_env(a.map,a.map_name,episode_ticks=2100)
    try:
        policy=DoomAgent(env.observation_space,env.action_space,'cpu',weights_path=a.weights,seed=a.policy_seed)
        obs,_=env.reset(seed=a.seed,options={'record_path':str(a.record.resolve())} if a.record else None);policy.reset()
        if a.show:import cv2
        done=False;steps=0
        while not done:
            if a.show:
                cv2.imshow('DOOM - learned PPO policy',cv2.resize(obs['rgb'][...,::-1],(800,600)))
                if cv2.waitKey(100)&255==ord('q'):break
            obs,_,term,trunc,info=env.step(policy.act(obs));steps+=1;done=term or trunc
        st=env.privileged_state()
        print({'decisions':steps,'exit':st['exit_reached'],'dead':st['dead'],'timeout':st['timeout'],'kills':st['kills']})
    finally:
        env.close()
        if a.show:cv2.destroyAllWindows()

if __name__=='__main__':main()
