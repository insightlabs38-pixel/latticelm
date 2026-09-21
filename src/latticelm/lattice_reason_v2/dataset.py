"""Compact token reservoir with stable-ID metadata regeneration."""
from __future__ import annotations
from collections import Counter
import hashlib,json,os,resource,shutil,time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from .core import GENERATOR_VERSION,generate_example,verify_example
from .audit import contamination_hits

SCHEMA="lattice-reason-v2-reservoir-v1"; MILESTONES=(150_000_000,300_000_000,500_000_000,1_000_000_000)
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def adaptive_target(tokens_per_second,bytes_per_token,available_bytes,seconds_to_cutoff,safety=1.35,reserve_bytes=20<<30):
 affordable=max(0,int((available_bytes-reserve_bytes)/max(bytes_per_token,1e-9)));timely=max(0,int(tokens_per_second*seconds_to_cutoff/safety));cap=min(affordable,timely,1_000_000_000)
 fits=[x for x in MILESTONES if x<=cap];return max(fits,default=150_000_000)
def build(output,tokenizer_path,target_tokens,seed=260921,shard_tokens=5_000_000,registry=None,stop=None,reserve_bytes=20<<30):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);tok=Tokenizer.from_file(str(tokenizer_path));tok_hash=sha256(tokenizer_path);state_path=output/"build_state.json"
 if shutil.disk_usage(output).free-target_tokens*4<reserve_bytes:raise RuntimeError("v2 reservoir would cross disk reserve")
 state=json.loads(state_path.read_text()) if state_path.exists() else {"schema":SCHEMA,"generator_version":GENERATOR_VERSION,"tokenizer_sha256":tok_hash,"seed":seed,"next_index":0,"tokens":0,"shards":[],"contamination_rejections":0}
 if state["tokenizer_sha256"]!=tok_hash or state["seed"]!=seed:raise RuntimeError("reservoir resume identity mismatch")
 started=time.monotonic();families=Counter();surfaces=Counter()
 while state["tokens"]<target_tokens and not(stop and stop()):
  if shutil.disk_usage(output).free<reserve_bytes:raise RuntimeError("v2 reservoir disk reserve crossed")
  ids=[];records=[];first=state["next_index"]
  while len(ids)<min(shard_tokens,target_tokens-state["tokens"]):
   ex=generate_example(state["next_index"],seed,"train",1+state["next_index"]%4)
   if not verify_example(ex):raise RuntimeError("generated example failed verifier")
   if contamination_hits(ex.prompt+(" "+ex.answer),registry or {}):state["next_index"]+=1;state["contamination_rejections"]=state.get("contamination_rejections",0)+1;continue
   encoded=tok.encode(ex.prompt+" "+ex.answer).ids[:256]
   if len(encoded)<2:state["next_index"]+=1;continue
   room=min(len(encoded),target_tokens-state["tokens"]-len(ids),shard_tokens-len(ids));offset=len(ids);ids.extend(encoded[:room]);records.append({"id":ex.global_id,"offset":offset,"length":room,"world_state_hash":ex.world_state_hash});families.update(ex.skills);surfaces[ex.surface_grammar]+=1;state["next_index"]+=1
  number=len(state["shards"]);partial=output/f".tokens-{number:05d}.partial";final=output/f"tokens-{number:05d}.int32";np.asarray(ids,dtype=np.int32).tofile(partial);os.replace(partial,final);meta=output/f"tokens-{number:05d}.ids.jsonl";meta.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in records));entry={"path":final.name,"sha256":sha256(final),"id_index":meta.name,"id_index_sha256":sha256(meta),"tokens":len(ids),"example_index_range":[first,state["next_index"]]};state["shards"].append(entry);state["tokens"]+=len(ids);tmp=state_path.with_suffix(".tmp");tmp.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n");os.replace(tmp,state_path)
 elapsed=time.monotonic()-started;manifest={**state,"status":"COMPLETE" if state["tokens"]>=target_tokens else "SAFE_STOP","target_tokens":target_tokens,"context_limit":256,"family_counts":dict(families),"surface_counts":dict(surfaces),"wall_seconds":elapsed,"tokens_per_second":state["tokens"]/max(elapsed,1e-9),"disk_bytes":sum((output/x["path"]).stat().st_size for x in state["shards"]),"free_disk_bytes":shutil.disk_usage(output).free,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};path=output/"manifest.json";path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return manifest
def verify_manifest(path,tokenizer_path):
 p=Path(path);x=json.loads(p.read_text())
 if x["schema"]!=SCHEMA or x["generator_version"]!=GENERATOR_VERSION or x["tokenizer_sha256"]!=sha256(tokenizer_path):raise ValueError("reservoir identity mismatch")
 if sum(s["tokens"] for s in x["shards"])!=x["tokens"]:raise ValueError("token count mismatch")
 for s in x["shards"]:
  if sha256(p.parent/s["path"])!=s["sha256"] or sha256(p.parent/s["id_index"])!=s["id_index_sha256"]:raise ValueError("shard hash mismatch")
 return x
