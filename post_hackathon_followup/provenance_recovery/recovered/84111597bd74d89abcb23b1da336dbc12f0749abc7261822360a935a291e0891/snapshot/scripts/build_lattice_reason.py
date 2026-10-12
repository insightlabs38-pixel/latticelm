"""Build a fixture, fixed milestone, validation suite, or adaptive canonical pool."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from latticelm.lattice_reason.dataset import build_canonical,build_validation

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--target",type=int)
    p.add_argument("--seed",type=int,default=7319);p.add_argument("--shard-tokens",type=int,default=1_000_000);p.add_argument("--adaptive",action="store_true");p.add_argument("--fixture-metadata",action="store_true")
    p.add_argument("--soft-hours",type=float,default=7.5);p.add_argument("--hard-hours",type=float,default=8.0);p.add_argument("--min-free-gib",type=float,default=5)
    p.add_argument("--validation",action="store_true");p.add_argument("--validation-per-cell",type=int,default=40)
    a=p.parse_args();commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    governed=["src/latticelm/lattice_reason","scripts/build_lattice_reason.py","tests/test_lattice_reason.py"]
    if subprocess.run(["git","diff-index","--quiet","HEAD","--",*governed],cwd=ROOT).returncode:raise SystemExit("canonical generation requires committed generator source")
    tokenizer=ROOT/"artifacts/tokenizers/babylm_2026_4k.json"
    if a.validation:result=build_validation(a.output.resolve(),tokenizer,a.validation_per_cell,a.seed)
    else:
        if a.target is None:p.error("--target is required unless --validation is used")
        result=build_canonical(a.output.resolve(),tokenizer,a.target,a.seed,a.shard_tokens,commit,a.soft_hours*3600,a.hard_hours*3600,a.adaptive,int(a.min_free_gib*(1<<30)),a.fixture_metadata)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
