#!/usr/bin/env python3
import argparse,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--audio-dir',required=True);a=p.parse_args()
manifest=ROOT/'config/narration.json';data=json.loads(manifest.read_text());source=Path(a.audio_dir)
for item in data:
    matches=[source/f"{item['scene']}{ext}" for ext in ('.wav','.mp3','.m4a')]
    found=next((x for x in matches if x.exists()),None)
    if not found:continue
    dest=ROOT/'assets/audio'/found.name;shutil.copy2(found,dest)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(dest)]))
    item['actual_audio']=str(dest.relative_to(ROOT));item['actual_duration']=float(probe['format']['duration'])
manifest.write_text(json.dumps(data,indent=2)+'\n');print('Imported',sum(bool(x.get('actual_audio')) for x in data),'clips')
