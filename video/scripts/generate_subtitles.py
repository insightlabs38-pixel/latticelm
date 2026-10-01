#!/usr/bin/env python3
"""Write an optional sidecar SRT without changing the master image."""
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TIMES={'Opening':8,'Architecture':74,'Training':56,'PostTraining':67,'Recovery':61,'FinalProof':17,'Recap':15}
def stamp(t):
    ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000);return f'{h:02}:{m:02}:{s:02},{ms:03}'
p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'renders/latticelm.srt'));p.add_argument('--timings');a=p.parse_args()
if a.timings:TIMES.update(json.loads(Path(a.timings).read_text()))
entries=[];start=0;index=1
for seg in json.loads((ROOT/'config/narration.json').read_text()):
    name=seg['scene'];length=min(TIMES[name],seg.get('actual_duration') or seg['estimated_duration'])
    words=seg['text'].split();chunks=[' '.join(words[i:i+9]) for i in range(0,len(words),9)]
    for j,chunk in enumerate(chunks):
        a0=start+length*j/len(chunks);b=start+length*(j+1)/len(chunks)
        entries.append(f'{index}\n{stamp(a0)} --> {stamp(b)}\n{chunk}\n');index+=1
    start+=TIMES[name]
Path(a.output).write_text('\n'.join(entries));print(a.output)
