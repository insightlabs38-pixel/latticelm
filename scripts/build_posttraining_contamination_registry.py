#!/usr/bin/env python3
"""Freeze a text-only registry; benchmark examples never become training rows."""
import argparse,json
from pathlib import Path
from datasets import load_dataset
from latticelm.lattice_reason_v2.audit import compile_registry
SPECS=(("Rowan/hellaswag",None),("allenai/ai2_arc","ARC-Easy"),("ybisk/piqa",None),("allenai/winogrande","winogrande_xl"),("Salesforce/wikitext","wikitext-103-raw-v1"))
def texts():
 for repo,config in SPECS:
  for split in ("train","validation","test"):
   try:rows=load_dataset(repo,config,split=split)
   except Exception:continue
   for row in rows:
    parts=[]
    for value in row.values():
     if isinstance(value,str):parts.append(value)
     elif isinstance(value,list):parts.extend(str(x) for x in value)
     elif isinstance(value,dict):parts.extend(str(x) for x in value.values())
    if parts:yield " ".join(parts)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();registry=compile_registry(texts());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(registry,separators=(",",":"))+"\n");print(json.dumps({"status":"PASS","exact":len(registry["exact"]),"grams":len(registry["grams"]),"output":str(a.output)}))
if __name__=="__main__":main()
