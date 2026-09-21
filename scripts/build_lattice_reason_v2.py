#!/usr/bin/env python3
import argparse,json,time,shutil
from pathlib import Path
from latticelm.lattice_reason_v2.dataset import adaptive_target,build
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT/"artifacts/posttraining/lattice_reason_v2");p.add_argument("--tokenizer",type=Path,default=ROOT/"artifacts/tokenizers/final_corpus_4k.json");p.add_argument("--registry",type=Path);p.add_argument("--target-tokens",type=int);p.add_argument("--measured-tps",type=float,default=50000);p.add_argument("--seconds-to-cutoff",type=float,default=24*3600);a=p.parse_args();target=a.target_tokens or adaptive_target(a.measured_tps,4.2,shutil.disk_usage(ROOT).free,a.seconds_to_cutoff);registry=json.loads(a.registry.read_text()) if a.registry else None;print(json.dumps(build(a.output,a.tokenizer,target,registry=registry),indent=2))
if __name__=="__main__":main()
