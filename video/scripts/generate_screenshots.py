#!/usr/bin/env python3
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STILLS=[('01_architecture','architecture','ArchitectureStill'),('02_data_pipeline','training','PipelineStill'),('03_optimizer','training','OptimizerStill'),('04_wsd_training','training','ScheduleStill'),('05_posttraining','posttraining','PostTrainingStill'),('06_recovery_evaluation','recovery','RecoveryStill'),('07_final_recap','recap','RecapStill')]
DESCRIPTIONS={
    '01_architecture':'Twelve-layer Co4 model, six query heads sharing two KV streams, dimensions and parameter allocation.',
    '02_data_pipeline':'DATA-D-v4 source processing, deduplication, tokenization and certified usable token count.',
    '03_optimizer':'Muon and AdamW parameter allocation with exact counts and percentages.',
    '04_wsd_training':'WSD schedule aligned with late validation improvement.',
    '05_posttraining':'Post-training method branches and capability gates.',
    '06_recovery_evaluation':'Retention, targeted proxy and full GIBC comparison for evaluated candidates.',
    '07_final_recap':'Architecture, training, evaluations, research finding and BASE selection.',
}
p=argparse.ArgumentParser();p.add_argument('--resolution',default='1920,1080');p.add_argument('--debug',action='store_true');p.add_argument('--require-final',action='store_true');p.add_argument('--also-1440',action='store_true');a=p.parse_args()
output=ROOT/'screenshots'/('debug' if a.debug else 'generated');output.mkdir(parents=True,exist_ok=True)
final_output=ROOT/'final/screenshots';final_output.mkdir(parents=True,exist_ok=True)
state=json.loads((ROOT/'config/metrics.json').read_text())['status']
if a.require_final and state!='final':raise SystemExit('Final screenshot export refused: selected results are still provisional')
env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
if a.debug:env['LATTICE_QA']='1'
manifest=[]
for filename,module,scene in STILLS:
    media=ROOT/'renders'/'screenshots'
    cmd=[sys.executable,'-m','manim','-s','-qk','-r',a.resolution,'--disable_caching','--media_dir',str(media),str(ROOT/'scenes'/f'{module}.py'),scene]
    subprocess.run(cmd,check=True,env=env,stdout=subprocess.DEVNULL)
    imgs=sorted(media.rglob(f'{scene}*.png'),key=lambda x:x.stat().st_mtime)
    if not imgs:raise RuntimeError(f'Missing {scene}')
    dest=output/f'{filename}.png';shutil.copy2(imgs[-1],dest)
    if not a.debug:shutil.copy2(dest,final_output/f'{filename}.png')
    entry={'filename':str(dest.relative_to(ROOT)),'scene':scene,'frame':'final static state','description':DESCRIPTIONS[filename],'final_results_required':filename in ('06_recovery_evaluation','07_final_recap'),'status':state,'resolution':a.resolution}
    if not a.debug and a.resolution=='1920,1080' and a.also_1440:
        high=ROOT/'screenshots/generated_1440'/f'{filename}.png';high.parent.mkdir(parents=True,exist_ok=True)
        high_cmd=cmd.copy();high_cmd[high_cmd.index('-r')+1]='2560,1440'
        subprocess.run(high_cmd,check=True,env=env,stdout=subprocess.DEVNULL)
        high_imgs=sorted(media.rglob(f'{scene}*.png'),key=lambda x:x.stat().st_mtime)
        shutil.copy2(high_imgs[-1],high)
        high_final=ROOT/'final/screenshots_1440'/f'{filename}.png';high_final.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(high,high_final)
        entry['high_resolution_filename']=str(high.relative_to(ROOT))
    manifest.append(entry)
(ROOT/'screenshots/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated',len(manifest),'screenshots in',output)
