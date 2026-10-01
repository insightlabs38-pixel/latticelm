#!/usr/bin/env python3
"""Render QA stills with safe-region overlays and inspect image bounds."""
import json,subprocess,sys,os
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env['LATTICE_STRICT']='1'
manifest_path=ROOT/'screenshots/manifest.json'
prior=manifest_path.read_text() if manifest_path.exists() else None
try:
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_screenshots.py'),'--debug'],check=True,env=env)
    debug_manifest=json.loads(manifest_path.read_text())
finally:
    if prior is not None: manifest_path.write_text(prior)
issues=[]
for item in debug_manifest:
    p=ROOT/item['filename'];im=Image.open(p)
    if im.size!=(1920,1080):issues.append(f'{p}: {im.size}')
if issues:raise SystemExit('\n'.join(issues))
print('Seven debug stills rendered at 1920×1080; inspect screenshots/debug/')
