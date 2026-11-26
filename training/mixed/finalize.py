"""Freeze selected validation winner, evaluate test once, wire inference weights.

This performs local artifact preparation only. No commit/push or external submission.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from training.mixed.evaluate import evaluate, ROOT
from training.eval_two_rooms import run_eval

def finalize(runs):
    out=ROOT/'docs/project/FINAL_RESULTS.json'
    if out.exists():raise FileExistsError('Final evaluation already exists; do not tune against test results.')
    candidates=[]
    for run in runs:
        s=json.loads((run/'selection.json').read_text())
        if len(s['curve'])!=10:raise ValueError(f'Expected 10 checkpoints: {run}')
        candidates.append(s['selected'])
    reference=json.loads((ROOT/'training/results/mixed_reference_validation.json').read_text())
    n=len(reference['episodes']);combat=reference['tasks']['combat']
    candidates.append(dict(checkpoint=reference['weights'],sha256=reference['sha256'],tasks=reference['tasks'],rank=[reference['balanced_exit_rate'],combat['kills']/combat['episodes'],-sum(r['decisions'] for r in reference['episodes'])/n,-1024000],decisions=1024000))
    selected=max(candidates,key=lambda r:r['rank'])
    w=Path(selected['checkpoint']);w=w if w.is_absolute() else ROOT/w
    if hashlib.sha256(w.read_bytes()).hexdigest()!=selected['sha256']:raise ValueError('Selected checkpoint changed')
    # Write immutable selection before observing test outcomes.
    frozen=ROOT/'docs/project/FROZEN_SELECTION.json'
    if frozen.exists():raise FileExistsError(frozen)
    frozen.write_text(json.dumps({'selected':selected,'candidates':candidates},indent=2)+'\n')
    test=evaluate(w,ROOT/'training/mixed/manifest.json',split='test')
    random_test=evaluate(None,ROOT/'training/mixed/manifest.json',split='test')
    official=run_eval(w,ROOT/'training/results/final_two_rooms')
    target=ROOT/'submission/weights/ppo_pilot.pt'
    backup=ROOT/'training/results/original_submission_pilot.pt'
    if not backup.exists():shutil.copy2(target,backup)
    shutil.copy2(w,target)
    payload={'selection_rule':'Highest equal-task validation exit rate; ties combat kills/episode, lower mean decisions, earlier checkpoint. Compare both mixed seeds and previous progression reference.','selected':selected,'candidates':candidates,'test':test,'random_test':random_test,'official_map':official,'test_limitations':'Navigation test geometries were evaluated in earlier Stage 3 experiments. Combat variants share those geometries. Progression test policy evaluation was held back until this freeze. Not organizer hidden-map performance.','submission_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    out.write_text(json.dumps(payload,indent=2)+'\n');return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs='+');a=p.parse_args();r=finalize(a.runs);print(json.dumps({'selected':r['selected'],'test':r['test']['tasks'],'official_map':r['official_map']},indent=2))
