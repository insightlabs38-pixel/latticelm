#!/usr/bin/env python3
"""Materialize exact DATA-D-v4 training-provenance transformations only."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from latticelm.lattice_reason_v2.natural import generate_from_tokens,verify
from latticelm.posttraining.state import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--examples",type=int,default=100000);a=p.parse_args();top=json.loads(a.manifest.read_text())
 if top["tokenizer_sha256"]!=sha256(a.tokenizer):raise RuntimeError("DATA-D/tokenizer identity mismatch")
 base=a.manifest.parent;arrays=[]
 # Deliberately consume only top-level training shards, never validation.
 for item in top["shards"]:
  meta=json.loads((base/item["manifest_path"]).read_text());arrays.append((meta["source"]+":"+meta["path"],np.memmap(base/meta["path"],dtype="<i4",mode="r")))
 a.output.mkdir(parents=True,exist_ok=True);path=a.output/"examples.jsonl";tmp=path.with_suffix(".tmp")
 with tmp.open("w") as f:
  for i in range(a.examples):
   source,array=arrays[i%len(arrays)];ex=generate_from_tokens(array,i,source_id=source)
   if not verify(ex,array):raise RuntimeError("natural provenance verification failed")
   f.write(json.dumps(ex.to_dict(),sort_keys=True)+"\n")
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);manifest={"schema":"lattice-reason-v2-natural-bank-v1","examples":a.examples,"data":path.name,"data_sha256":sha256(path),"source_manifest_sha256":sha256(a.manifest),"tokenizer_sha256":sha256(a.tokenizer),"training_shards_only":True,"transformations":["source_contiguous_next_span"],"verification":"PASS"};(a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest))
if __name__=="__main__":main()
