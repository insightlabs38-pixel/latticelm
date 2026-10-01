#!/usr/bin/env python3
import argparse,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
p=argparse.ArgumentParser();p.add_argument('--tape',choices=['verification','evaluation','inference'],default='evaluation');a=p.parse_args()
env=os.environ.copy();env['PATH']=str(ROOT/'tools')+':'+env['PATH']
subprocess.run([str(ROOT/'tools/vhs'),str(ROOT/'vhs'/f'{a.tape}.tape')],cwd=ROOT,env=env,check=True)
