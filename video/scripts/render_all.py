#!/usr/bin/env python3
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAP={'Opening':'opening','Architecture':'architecture','Training':'training','PostTraining':'posttraining','Recovery':'recovery','FinalProof':'final_proof','Recap':'recap'}
p=argparse.ArgumentParser();p.add_argument('--quality',choices=['preview','final'],default='final');p.add_argument('--only',choices=list(MAP));a=p.parse_args()
media=ROOT/'renders'/a.quality;media.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
index=media/'scene_paths.json'
outputs=json.loads(index.read_text()) if a.only and index.exists() else {}
for name,file in MAP.items():
    if a.only and name!=a.only:continue
    cmd=[sys.executable,'-m','manim','-q'+('l' if a.quality=='preview' else 'h'),'--disable_caching','--fps','60' if a.quality=='final' else '15','--media_dir',str(media),str(ROOT/'scenes'/f'{file}.py'),name]
    subprocess.run(cmd,check=True,env=env)
    matches=sorted(media.rglob(f'{name}.mp4'),key=lambda p:p.stat().st_mtime)
    if not matches:raise RuntimeError(f'No render for {name}')
    outputs[name]=str(matches[-1])
(media/'scene_paths.json').write_text(json.dumps(outputs,indent=2)+'\n')
