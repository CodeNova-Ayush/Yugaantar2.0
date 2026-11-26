from training.task_rewards import TaskRewardTracker

def state(kills=0, damage=0., x=32., y=32.):
    return dict(kills=kills, damage_taken=damage, x=x,y=y)

def test_events_paid_once_and_reset():
    t=TaskRewardTracker(state())
    assert t.step(state(1,10.,96.), .3,.002,.01)==dict(kill_bonus=.3,damage_cost=-.02,exploration_bonus=.01)
    assert sum(t.step(state(1,10.,96.),.3,.002,.01).values())==0
    assert sum(t.step(state(1,10.,32.),.3,.002,.01).values())==0
    t=TaskRewardTracker(state())
    assert t.step(state(1),.3,.002,.01)['kill_bonus']==.3

def test_negative_coordinate_cells():
    assert TaskRewardTracker.cell(state(x=-1))==(-1,0)

def test_mixed_manifest_split_and_sampling():
    import json
    from pathlib import Path
    import pytest
    maps=json.loads(Path('training/mixed/manifest.json').read_text())['maps']
    for task,expected in [('navigation',.3),('progression',.4),('combat',.3)]:
        assert sum(m['sampling_weight'] for m in maps if m['task']==task and m['split']=='train')==pytest.approx(expected)
    hashes={split:{m['sha256'] for m in maps if m['split']==split} for split in ['train','val','test']}
    assert not hashes['train'] & (hashes['val'] | hashes['test'])
    assert not hashes['val'] & hashes['test']
