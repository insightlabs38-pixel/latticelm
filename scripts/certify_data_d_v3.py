"""Independent acceptance gate and report writer for DATA-D-BROAD-v3."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from latticelm.data_d import Int32Shard,ExactMixtureV2,SourceStream,sha256_file,verify_top_manifest
ROOT=Path(__file__).resolve().parents[1];TOK=ROOT/"artifacts/tokenizers/babylm_2026_4k.json"
def certify(path:Path,minimum=1_000_000_000):
 top=verify_top_manifest(path,TOK);base=path.parent
 train=set();val=set();arrays={s:[] for s in ("fineweb_edu","wikipedia","fineweb")};source_tokens={s:0 for s in arrays}
 for child in top["shards"]:
  m=json.loads((base/child["manifest_path"]).read_text());train.update(m["stable_document_ids"]);source_tokens[m["source"]]+=m["token_count"];arrays[m["source"]].append(Int32Shard(base/m["path"],m).tokens)
 for child in top["validation_shards"]:
  if sha256_file(base/child["manifest_path"])!=child["manifest_sha256"]:raise ValueError("validation child hash mismatch")
  m=json.loads((base/child["manifest_path"]).read_text());val.update(m["stable_document_ids"]);Int32Shard(base/m["path"],m)
 if train&val:raise ValueError("train/validation overlap")
 if top["total_unique_tokens"]<minimum:raise ValueError("less than one billion clean tokens")
 if len(train)!=top["total_documents"]:raise ValueError("duplicate train IDs")
 mix=ExactMixtureV2({s:SourceStream(arrays[s],128,920+i) for i,s in enumerate(arrays)});x,y,_=mix.batch();digest=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest();saved=mix.state_dict();x2,y2,_=mix.batch();expected=hashlib.sha256(x2.tobytes()+y2.tobytes()).hexdigest();mix.load_state_dict(saved);a,b,_=mix.batch()
 if hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()!=expected:raise ValueError("exact resume failure")
 commit=top["builder_git_commit"];source=subprocess.check_output(["git","show",f"{commit}:scripts/build_data_d_v3.py"],cwd=ROOT)
 report={"acceptance":"PASS","clean_unique_train_tokens":top["total_unique_tokens"],"train_documents":len(train),"validation_documents":len(val),"source_tokens":source_tokens,"source_shares":{s:source_tokens[s]/top["total_unique_tokens"] for s in source_tokens},"manifest_sha256":sha256_file(path),"builder_commit":commit,"builder_source_sha256":hashlib.sha256(source).hexdigest(),"train_validation_disjointness":"PASS","all_shards_mmap_and_hash":"PASS","deterministic_first_batch_sha256":digest,"exact_resume":"PASS","stable_unique_ids":"PASS","overlap_handling":top["overlap_registry"],"decontamination":top["decontamination_version"]}
 (base/"certification.json").write_text(json.dumps(report,indent=2)+"\n");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("manifest",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--minimum",type=int,default=1_000_000_000);a=p.parse_args();r=certify(a.manifest,a.minimum);a.output.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))
if __name__=="__main__":main()
