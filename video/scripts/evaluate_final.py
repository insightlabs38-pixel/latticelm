#!/usr/bin/env python3
"""Designed, evidence-backed terminal readout for VHS."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'config/metrics.json').read_text());f=d['final']
verification='PROVISIONAL'
if d['status']=='final':
    source=d.get('source') or {}
    selection_path=Path(source.get('selection',''))
    selection=json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    checkpoint=Path(selection.get('checkpoint',''))
    if not checkpoint.is_file(): sys.exit('Selected checkpoint file missing; cannot verify')
    digest=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    verification='PASS' if digest.hexdigest()==f['final_checkpoint_sha'] else 'FAIL'
fmt=lambda v,spec='.3f': 'PENDING' if v is None else format(v,spec)
pct=lambda v:'PENDING' if v is None else f'{v*100:.1f}%'
print('LatticeLM  /  final verification\n')
if '--verification-only' in sys.argv:
    for k,v in [('checkpoint',f['final_model_name'] or 'PENDING'),('parameters',f"{json.loads((ROOT/'data/architecture.json').read_text())['parameters']:,}"),('checkpoint hash',(f['final_checkpoint_sha'] or 'PENDING')[:12]),('integrity',verification)]:print(f'{k:<18}{v}')
    if d['status']=='final' and verification!='PASS':sys.exit('Checkpoint hash mismatch')
    sys.exit(0)
for k,v in [('checkpoint',f['final_model_name'] or 'PENDING'),('parameters',f"{json.loads((ROOT/'data/architecture.json').read_text())['parameters']:,}"),('DATA-D',fmt(f['data_d'])),('WikiText BPB',fmt(f['wikitext_bpb'])),('ARC-Easy',pct(f['arc_easy'])),('Reasoning / 1024',pct(f['reasoning_accuracy'])),('checkpoint hash',(f['final_checkpoint_sha'] or 'PENDING')[:12]),('verification',verification)]:print(f'{k:<18}{v}')
if d['status']=='final' and verification!='PASS':sys.exit('Checkpoint hash mismatch')
if d['status']=='final' and not (f['final_checkpoint_sha'] and f['data_d'] is not None and f['reasoning_accuracy'] is not None):sys.exit('Final evidence incomplete')
