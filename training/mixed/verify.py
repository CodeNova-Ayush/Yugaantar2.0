"""Verify combat map loading, enemy counts, and geometric exit connectivity."""
import json
from collections import deque
from pathlib import Path
from doom_env import make_env
from training.mixed.generate import ROOT

def verify():
    rows=[]
    for m in json.loads((ROOT/'training/mixed/manifest.json').read_text())['maps']:
        if m['task']!='combat':continue
        grid=m['grid'];start=next((r,c) for r,row in enumerate(grid) for c,ch in enumerate(row) if ch=='P')
        seen={start};q=deque([start])
        while q:
            r,c=q.popleft()
            for y,x in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0<=y<len(grid) and 0<=x<len(grid[0]) and grid[y][x]!='#' and (y,x) not in seen:
                    seen.add((y,x));q.append((y,x))
        assert any(grid[r][c]=='E' for r,c in seen), m['name']
        env=make_env(ROOT/m['wad_path'])
        try:
            obs,_=env.reset(seed=0)
            assert env.observation_space.contains(obs)
            count=sum(o.name in ('Zombieman','DoomImp') for o in env._game.get_state().objects)
            assert count==m['enemies'],(m['name'],count)
            rows.append(dict(layout=m['name'],enemies=count,geometric_exit_reachable=True))
        finally:env.close()
    out=ROOT/'training/results/mixed_map_verification.json';out.write_text(json.dumps(rows,indent=2)+'\n')
    print(f'{len(rows)} combat layouts verified; this checks geometry/spawns, not learned success.')

if __name__=='__main__':verify()
