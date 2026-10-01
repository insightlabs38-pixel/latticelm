#!/usr/bin/env python3
"""Synchronize independently rendered Manim clips, audio and optional VHS evidence."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from components.timing import allocate
from config.timings import SCENES
TIMES=dict(SCENES)
MIN_HOLDS={'Opening':7,'Architecture':78,'Training':67,'PostTraining':40,'Recovery':40,'FinalProof':10,'Recap':22}
p=argparse.ArgumentParser();p.add_argument('--quality',choices=['preview','final'],default='final');p.add_argument('--output');p.add_argument('--subtitles',action='store_true');p.add_argument('--vhs',default=str(ROOT/'assets/terminal/evaluation.mp4'));a=p.parse_args()
paths=json.loads((ROOT/'renders'/a.quality/'scene_paths.json').read_text());narr={x['scene']:x for x in json.loads((ROOT/'config/narration.json').read_text())}
state=json.loads((ROOT/'config/metrics.json').read_text())['status']
default_name=f"latticelm_{a.quality}_provisional.mp4" if state!='final' else f"latticelm_{a.quality}.mp4"
out=Path(a.output or ROOT/'renders'/default_name);out.parent.mkdir(parents=True,exist_ok=True)
# Audio adds time at its own scene; recover that time from other scene-ending holds.
alloc=allocate(TIMES,MIN_HOLDS,{k:v.get('actual_duration') for k,v in narr.items()})
(ROOT/'renders'/a.quality/'composition_timings.json').write_text(json.dumps(alloc,indent=2)+'\n')
parts=[]
for name,target in alloc.items():
    source=paths[name];item=narr.get(name,{})
    audio=ROOT/item['actual_audio'] if item.get('actual_audio') else None
    part=ROOT/'renders'/a.quality/f'composed_{name}.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-i',source]
    overlay=False
    if overlay:cmd+=['-i',a.vhs]
    if audio:cmd+=['-i',str(audio)]
    else:cmd+=['-f','lavfi','-i','anullsrc=channel_layout=stereo:sample_rate=48000']
    filterv=f'[0:v]scale=1920:1080:flags=lanczos,fps=60,tpad=stop_mode=clone:stop_duration={target},trim=duration={target},setpts=PTS-STARTPTS[v0]'
    if overlay:
        filterv+=f';[1:v]scale=530:328:force_original_aspect_ratio=decrease,pad=530:334:(ow-iw)/2:(oh-ih)/2:color=0x101923,trim=duration={target},tpad=stop_mode=clone:stop_duration={target}[term];[v0][term]overlay=x=1235:y=389:shortest=1[v]'
    else:filterv+=';[v0]null[v]'
    cmd+=['-filter_complex',filterv,'-map','[v]']
    if audio:
        idx=2 if overlay else 1
        cmd+=['-map',f'{idx}:a:0','-af',f'apad,atrim=duration={target}','-c:a','aac','-b:a','192k']
    else:cmd+=['-map',f'{2 if overlay else 1}:a:0','-t',str(target),'-c:a','aac','-b:a','128k']
    cmd+=['-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-r','60','-t',str(target),str(part)]
    subprocess.run(cmd,check=True);parts.append(part)
concat=ROOT/'renders'/a.quality/'concat.txt';concat.write_text(''.join(f"file '{x.resolve()}'\n" for x in parts))
subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(out)],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/validate_runtime.py'),str(out)],check=True)
if a.subtitles:
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_subtitles.py'),'--output',str(out.with_suffix('.srt')),'--timings',str(ROOT/'renders'/a.quality/'composition_timings.json')],check=True)
print(out)
