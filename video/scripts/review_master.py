#!/usr/bin/env python3
"""Sample the composed master at scene boundaries and regular intervals."""
import argparse,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('video');a=p.parse_args()
out=ROOT/'renders/qa';out.mkdir(parents=True,exist_ok=True)
times=[0,5,9,40,81,84,110,137,141,175,205,209,238,265,270,282,285,297]
frames=[]
for i,t in enumerate(times):
    path=out/f'master_{i:02}.png'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(t),'-i',a.video,'-frames:v','1','-vf','scale=480:270',str(path)],check=True)
    im=Image.open(path).convert('RGB');ImageDraw.Draw(im).text((8,8),f'{t}s',fill='white');frames.append(im)
sheet=Image.new('RGB',(480*3,300*6),'#0b1017')
for i,im in enumerate(frames):sheet.paste(im,((i%3)*480,(i//3)*300))
sheet.save(out/'master_contact.png');print(out/'master_contact.png')
