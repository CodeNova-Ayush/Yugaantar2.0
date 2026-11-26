import json
from pathlib import Path
from training.curriculum.verify_curriculum import canonical_d4_signature

def test_mixed_geometry_splits():
    maps=json.loads(Path('training/mixed/manifest.json').read_text())['maps']
    seen={}
    for m in maps:
        sig=canonical_d4_signature(m['grid'])
        assert sig not in seen or seen[sig]==m['split']
        seen[sig]=m['split']

def test_default_inference_has_no_training_dependencies():
    import ast
    for path in [Path('submission/agent.py'),Path('submission/model.py')]:
        tree=ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):assert not (n.module or '').startswith('training')
            if isinstance(n,ast.Import):assert all(not a.name.startswith('training') for a in n.names)
