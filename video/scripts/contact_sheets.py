#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--quality',choices=['preview','final'],default='preview');a=p.parse_args()
paths=json.loads((ROOT/'renders'/a.quality/'scene_paths.json').read_text());out=ROOT/'renders/qa';out.mkdir(parents=True,exist_ok=True)
BEATS={'Opening':[1.8,4.6,5.4], 'Architecture':[2,9,29,49,63,77], 'Training':[1.2,17.3,34,49,67], 'PostTraining':[1.3,19.8,21], 'Recovery':[1.3,14.1,18,22], 'FinalProof':[1.2,3.2,4], 'Recap':[.7,3.5,4.8,5.5]}
for name,path in paths.items():
    duration=float(json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',path]))['format']['duration'])
    thumbs=[]
    for i in range(11):
        at=max(0,min(duration-.25,duration*i/10))
        tmp=out/f'{name}_{i:02}.png'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(at),'-i',path,'-frames:v','1','-vf','scale=480:270',str(tmp)],check=True)
        im=Image.open(tmp).convert('RGB');ImageDraw.Draw(im).text((8,8),f'{i*10}%',fill='white');thumbs.append(im)
    sheet=Image.new('RGB',(480*3,300*4),'#0b1017')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*480,(i//3)*300))
    sheet.save(out/f'{name}_contact.png')
    beat_frames=[]
    for beat in BEATS.get(name,[]):
        for side,at in [('before',max(0,beat-.15)),('after',min(duration-.25,beat+.15))]:
            tmp=out/f'{name}_beat_{beat:g}_{side}.png'
            subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(at),'-i',path,'-frames:v','1','-vf','scale=480:270',str(tmp)],check=True)
            im=Image.open(tmp).convert('RGB');ImageDraw.Draw(im).text((8,8),f'{beat:g}s {side}',fill='white');beat_frames.append(im)
    if beat_frames:
        columns=2;rows=(len(beat_frames)+1)//2;bs=Image.new('RGB',(960,rows*300),'#0b1017')
        for i,im in enumerate(beat_frames):bs.paste(im,((i%2)*480,(i//2)*300))
        bs.save(out/f'{name}_transitions.png')
print('Contact sheets:',out)
