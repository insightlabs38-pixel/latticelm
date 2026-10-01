import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent

def read(path):
    import os
    override=os.getenv('LATTICE_RECOVERY_DATA_PATH') if path=='data/recovery.json' else None
    return json.loads(Path(override).read_text()) if override else json.loads((ROOT/path).read_text())

def metric_state():
    import os
    override=os.getenv('LATTICE_METRICS_PATH')
    return json.loads(Path(override).read_text()) if override else read('config/metrics.json')

def provisional():
    return read('config/provisional_metrics.json')

def final():
    return metric_state()['final']

def is_provisional():
    return metric_state()['status']!='final'

def display(v,fmt='.3f',missing='PENDING'):
    return missing if v is None else format(v,fmt)

def pct(v,dec=1):
    return 'PENDING' if v is None else f'{v*100:.{dec}f}%' if v<=1 else f'{v:.{dec}f}%'
