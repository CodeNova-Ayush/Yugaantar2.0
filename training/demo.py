"""Run the submitted agent on a map; optionally display RGB and record a replay.
Usage: python -m training.demo --show --record /tmp/demo.lmp
"""
import argparse
import time
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
        if a.show:
            # Window visibility is a pre-init ViZDoom option. Reinitialize only
            # this demo's engine, before reset creates any geometry/sensor state.
            # Keep the official resolution, observations, buttons and tick rate.
            env._game.close()
            env._game.set_window_visible(True)
            env._game.init()
        policy=DoomAgent(env.observation_space,env.action_space,'cpu',weights_path=a.weights,seed=a.policy_seed)
        obs,_=env.reset(seed=a.seed,options={'record_path':str(a.record.resolve())} if a.record else None);policy.reset()
        done=False;steps=0
        while not done:
            started=time.monotonic()
            obs,_,term,trunc,info=env.step(policy.act(obs));steps+=1;done=term or trunc
            if a.show:
                time.sleep(max(0., 4 / 35 - (time.monotonic() - started)))
        st=env.privileged_state()
        print({'decisions':steps,'exit':st['exit_reached'],'dead':st['dead'],'timeout':st['timeout'],'kills':st['kills']})
    except KeyboardInterrupt:
        print('Demo stopped. This interrupted episode is not a measured win or loss.')
    finally:
        env.close()

if __name__=='__main__':main()
