"""Bounded expanded-map experiment; never overwrites submission weights or reads test results."""
import json, hashlib, shutil
from pathlib import Path
from training.procedural import generate as gen
from training.procedural.evaluate import evaluate
from training.ppo import PPOConfig
from training.train_ppo import train
from doom_env.wad import Lump, write_wad
from doom_env import make_env

ROOT=gen.ROOT
OUT=ROOT/'training/results/expanded_2m_s43'
MAN=ROOT/'training/expanded/manifest.json'
TASKS=list(gen.TASK_WEIGHTS)

def generate():
    if MAN.exists():return json.loads(MAN.read_text())
    seen=set()
    for path in [ROOT/'training/procedural/manifest.json',ROOT/'training/mixed/manifest.json',ROOT/'training/mixed_v2/manifest.json']:
        if path.exists():
            for m in json.loads(path.read_text())['maps']:seen.add(gen.d4_canonical(m['grid']))
    maps=[];folder=MAN.parent/'maps';folder.mkdir(exist_ok=True)
    for split,count,base in [('train',50,300000),('val',6,600000)]:
        for t,task in enumerate(TASKS):
            accepted=0
            for attempt in range(10000):
                seed=base+t*20000+attempt
                candidate=gen._generate_candidate(task,0 if accepted<count//2 else 1,seed)
                if candidate is None:continue
                grid,angle,things,meta=candidate;sig=gen.d4_canonical(grid)
                if sig in seen:continue
                seen.add(sig);name=f'expanded_{split}_{task}_{accepted:03}'
                path=folder/(name+'.wad')
                text=gen.procedural_grid_to_udmf(grid,player_angle=angle,extra_things=things)
                write_wad(path,[Lump('MAP01',b''),Lump('TEXTMAP',text.encode()),Lump('ENDMAP',b'')])
                env=make_env(path,episode_ticks=2100,render_buffers=False)
                try:
                    obs,_=env.reset(seed=0)
                    assert env.observation_space.contains(obs)
                finally:env.close()
                m=dict(name=name,split=split,task=task,map_name='MAP01',seed=seed,grid=grid,params=meta,player_angle=angle,wad_path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sampling_weight=gen.TASK_WEIGHTS[task]/count)
                for k in ['key_color','enemies','has_door']:
                    if k in meta:m[k]=meta[k]
                maps.append(m);accepted+=1
                if accepted==count:break
            assert accepted==count
    result=dict(generator_version='expanded_proc_v1',maps=maps,note='New seeds; geometry excluded against prior manifests. 200 train, 24 validation; no final test generated or used.')
    MAN.write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir()
    generate()
    rule={'rule':'Keep default unless balanced validation exits improve AND no task drops by more than 5 percentage points against default. Rank eligible candidates by balanced exits, combat kills, earlier checkpoint. No automatic replacement.','budget':2048000,'seed':43,'checkpoint_every':102400,'episodes_per_layout':20}
    (OUT/'protocol.json').write_text(json.dumps(rule,indent=2)+'\n')
    default=ROOT/'submission/weights/ppo_pilot.pt';shutil.copy2(default,OUT/'baseline.pt')
    baseline=evaluate(default,MAN,episodes=20)
    (OUT/'baseline_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    cfg=PPOConfig(total_decisions=2048000,seed=43,obs_version='v2',env_cache_size=16,key_reward=.25,door_passable_reward=.5,unlocked_door_reward=.2,kill_reward=.3,damage_penalty=.002,exploration_reward=.01)
    from dataclasses import replace
    train(replace(cfg,total_decisions=2048),MAN,OUT/'smoke',None)
    train(cfg,MAN,OUT/'training',None,checkpoint_every=102400)
    curve=[]
    for w in sorted((OUT/'training/checkpoints').glob('ckpt_*.pt'),key=lambda p:int(p.stem.split('_')[1])):
        r=evaluate(w,MAN,episodes=20);n=int(w.stem.split('_')[1])
        (OUT/f'validation_{n}.json').write_text(json.dumps(r,indent=2)+'\n')
        eligible=r['balanced_exit_rate']>baseline['balanced_exit_rate'] and all(r['tasks'][t]['exit_rate']>=baseline['tasks'][t]['exit_rate']-.05 for t in TASKS)
        row=dict(checkpoint=str(w),decisions=n,eligible=eligible,balanced_exit_rate=r['balanced_exit_rate'],tasks=r['tasks']);curve.append(row)
        print(json.dumps(row),flush=True)
    eligible=[r for r in curve if r['eligible']]
    winner=max(eligible,key=lambda r:(r['balanced_exit_rate'],r['tasks']['combat']['kills'],-r['decisions'])) if eligible else None
    (OUT/'recommendation.json').write_text(json.dumps(dict(recommended=winner,curve=curve,default_unchanged=True),indent=2)+'\n')
    print('COMPLETE. Default weights unchanged.',flush=True)

if __name__=='__main__':main()
