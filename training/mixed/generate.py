"""Deterministic combat variants and a weighted, split-preserving mixed manifest."""
import hashlib
import json
from pathlib import Path
from doom_env.wad import Lump, write_wad
from training.generate_maps import grid_to_udmf

ROOT = Path(__file__).resolve().parents[2]

def generate():
    out = ROOT / 'training/mixed'
    (out / 'maps').mkdir(parents=True, exist_ok=True)
    nav = json.loads((ROOT / 'training/curriculum/manifest.json').read_text())['maps']
    prog = json.loads((ROOT / 'training/progression/manifest.json').read_text())['maps']
    maps = [{**m, 'task': 'navigation'} for m in nav] + [{**m, 'task': 'progression'} for m in prog]
    for i, source in enumerate(nav):
        rows = source['grid']; h = len(rows)
        floors = [(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == '.']
        player = next((r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch == 'P')
        candidates = sorted(floors, key=lambda rc: (abs(rc[0]-player[0])+abs(rc[1]-player[1]),rc))
        candidates = [rc for rc in candidates if abs(rc[0]-player[0])+abs(rc[1]-player[1]) >= 2]
        if not candidates: raise ValueError(source['name'])
        # One or two low-tier enemies. Distinct placements, ammo, health; no spawn overlap.
        enemies = candidates[:1] if i % 2 == 0 else candidates[:2]
        things = [(r,c,3004 if j == 0 else 3001) for j,(r,c) in enumerate(enemies)]
        spare = [rc for rc in floors if rc not in enemies]
        things += [(r,c,kind) for (r,c),kind in zip(spare, [2007,2011])]
        text = grid_to_udmf('\n'.join(rows), player_angle=source['player_angle'])
        for r,c,kind in things:
            text += f'thing {{ x = {(c+.5)*64}; y = {(h-r-.5)*64}; type = {kind}; angle = 180; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}\n'
        name = 'combat_' + source['name']; path = out / 'maps' / (name+'.wad')
        write_wad(path, [Lump('MAP01',b''),Lump('TEXTMAP',text.encode()),Lump('ENDMAP',b'')])
        maps.append({**source, 'name':name, 'task':'combat', 'wad_path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'enemies':len(enemies)})
    weights = {'navigation':.3,'progression':.4,'combat':.3}
    for m in maps:
        count = sum(x['task']==m['task'] and x['split']==m['split'] for x in maps)
        m['sampling_weight'] = weights[m['task']]/count
    manifest = {'task_probabilities':weights, 'note':'Combat geometry derives only from the corresponding navigation split. No cross-split reuse.', 'maps':maps}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__ == '__main__':
    m=generate();print('Generated',len(m['maps']),'mixed layouts')
