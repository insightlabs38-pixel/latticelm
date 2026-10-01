#!/usr/bin/env python3
import argparse,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--max',type=float,default=300);a=p.parse_args()
d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',a.path]))['format']['duration'];d=float(d)
print(f'{a.path}: {d:.3f}s / limit {a.max:.3f}s')
if d>=a.max:sys.exit(1)
