import json
import os
import sys
from pathlib import Path

BASE_DIR = Path('reports/baselines')
cur_path = BASE_DIR / 'current.json'
cand_path = BASE_DIR / 'candidate.json'

if not cur_path.exists() or not cand_path.exists():
    print('Baseline files missing.')
    sys.exit(1)

with cur_path.open() as f:
    current = json.load(f)
with cand_path.open() as f:
    candidate = json.load(f)

regressions = []
for task, cand_metrics in candidate.get('hard', {}).items():
    cur_metrics = current.get('hard', {}).get(task, {})
    for key in ['auroc', 'auprc']:
        c_val = cand_metrics.get(key)
        b_val = cur_metrics.get(key)
        if c_val is None or b_val is None:
            continue
        if b_val - c_val > 0.03:
            regressions.append((task, key, b_val, c_val))

if regressions:
    for task, metric, old, new in regressions:
        print(f'{task} {metric} decreased from {old:.4f} to {new:.4f}')
    if os.getenv('ALLOW_REGRESSION') == '1':
        print('ALLOW_REGRESSION=1, ignoring regression')
        sys.exit(0)
    sys.exit(1)
else:
    print('No regression detected.')
    sys.exit(0)
