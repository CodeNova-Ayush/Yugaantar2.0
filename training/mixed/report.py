"""Render a concise Markdown report from frozen final results."""
import json
from pathlib import Path
from training.mixed.evaluate import ROOT

def report():
    r=json.loads((ROOT/'docs/project/FINAL_RESULTS.json').read_text())
    lines=['# Final measured results','', 'Selected checkpoint: `'+r['selected']['checkpoint']+'`.', '', 'SHA-256: `'+r['submission_sha256']+'`.', '', '| Task | Selected validation exits | Final test exits | Random test exits |','|---|---:|---:|---:|']
    for name in ['navigation','progression','combat']:
        v=r['selected']['tasks'][name];t=r['test']['tasks'][name];b=r['random_test']['tasks'][name]
        lines.append(f"| {name} | {v['exit']}/{v['episodes']} | {t['exit']}/{t['episodes']} | {b['exit']}/{b['episodes']} |")
    lines += ['',r['test_limitations'],'', '## Supplied map', '', json.dumps({k:r['official_map'][k] for k in ['episodes','exits','deaths','timeouts','kills_mean','door_passable_count']},indent=2),'','## Training evidence','']
    for run in ['mixed_1m_s42','mixed_1m_s43']:
        s=json.loads((ROOT/'training/results'/run/'summary.json').read_text())
        lines.append(f"- {run}: {s['decisions_done']:,} decisions; {s['completed_episodes']:,} completed episodes; {s['completed_exits']:,} training exits; {s['wall_seconds']:.1f} seconds. Training exits are on-policy experience, not held-out evaluation.")
    lines += ['', '## Limits and verification', '', 'Two mixed training seeds were used. This is not a large-seed statistical study. Validation selection used ten checkpoints per seed plus the earlier progression reference. Test data was not used to pick the checkpoint. Combat maps contain one or two low-tier enemies and are substantially simpler than an unknown gauntlet.', '', 'See validator_output.txt for the actual integration run and verification.json for test counts. MacOS CPU pinning/RAM warnings and any missing Linux verification must remain disclosed. No hidden-map win rate is claimed.']
    (ROOT/'docs/project/RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':report()
