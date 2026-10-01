#!/usr/bin/env python3
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAP={'Opening':'opening','Architecture':'architecture','Training':'training','PostTraining':'posttraining','Recovery':'recovery','FinalProof':'final_proof','Recap':'recap'}
p=argparse.ArgumentParser();p.add_argument('scene',choices=list(MAP));p.add_argument('--quality',choices=['l','h'],default='l');a=p.parse_args()
env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
cmd=[sys.executable,'-m','manim',f'-q{a.quality}','--disable_caching','--media_dir',str(ROOT/'renders/preview'),str(ROOT/'scenes'/f'{MAP[a.scene]}.py'),a.scene]
subprocess.run(cmd,env=env,check=True)
