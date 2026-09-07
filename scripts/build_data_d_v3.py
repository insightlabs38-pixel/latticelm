"""Restartable, globally deduplicated DATA-D-BROAD-v3 materializer."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sqlite3,subprocess,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from datasets import load_dataset
from latticelm.data_d import (CORPUS_V3_ID,DEDUP_VERSION,DECONTAMINATION_VERSION,SCHEMA_VERSION,
 ContaminationRegistry,Document,canonical_json,document_hash,hamming,normalize_text,normalized_words,
 paragraph_hashes,sha256_bytes,sha256_file,simhash64,validation_assignment,VALIDATION_ASSIGNMENT_VERSION)
from latticelm.tokenizer import load_tokenizer
try:
 from scripts.prepare_phase7a_data import reference_documents
 from scripts.prepare_phase7f_v2 import REVISIONS,LICENSES,common_validation_references
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
 from prepare_phase7a_data import reference_documents
 from prepare_phase7f_v2 import REVISIONS,LICENSES,common_validation_references

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";TOKENIZER=ART/"tokenizers/babylm_2026_4k.json"
OLD=ART/"data/phase7f_v2r1/canonical-100m";DEFAULT_OUT=ART/"data/data_d_v3/canonical-1b"
SOURCES=("fineweb_edu","wikipedia","fineweb");RESERVE=15*1024**3;SEED=99173

def atomic(path:Path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name("."+path.name+".tmp")
 with tmp.open("w") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def rows(source):
 repo,config,revision=REVISIONS[source];stream=load_dataset(repo,config,split="train",streaming=True,revision=revision)
 if source!="wikipedia":stream=stream.shuffle(seed=SEED+SOURCES.index(source),buffer_size=10_000)
 for raw in stream:
  text=normalize_text(raw.get("text",""));words=normalized_words(text)
  if len(words)<50 or (source!="wikipedia" and float(raw.get("language_score",1))<.95):yield None;continue
  if source=="fineweb_edu" and float(raw.get("int_score",0))<3:yield None;continue
  if source=="fineweb" and (len(words)<100 or (not set(words)&{"how","step","guide","explain","because","use","method"} and len(words)<300)):yield None;continue
  ident=str(raw.get("id"));meta={k:raw.get(k) for k in ("id","url","dump","date","title","file_path","language_score","int_score") if k in raw}
  yield Document(source,f"{source}:{ident}",text,meta,LICENSES[source])

class Index:
 def __init__(self,path):self.db=sqlite3.connect(path);self.db.execute("pragma journal_mode=WAL");self.db.execute("pragma synchronous=FULL")
 def check(self,doc):
  exact=document_hash(doc.text);row=self.db.execute("select id from exact where h=?",(exact,)).fetchone()
  if row:return "exact_document",row[0]
  for value in paragraph_hashes(doc.text):
   row=self.db.execute("select id from para where h=?",(value,)).fetchone()
   if row:return "exact_paragraph",row[0]
  fp=simhash64(doc.text);candidates={}
  for band in range(4):
   for ident,value in self.db.execute("select id,s from band where b=? and k=?",(band,(fp>>(16*band))&65535)):candidates[ident]=value&((1<<64)-1)
  close=sorted((hamming(fp,v),k) for k,v in candidates.items())
  return ("near_duplicate",close[0][1]) if close and close[0][0]<=3 else (None,None)
 def add_many(self,docs):
  with self.db:
   for doc in docs:
    exact=document_hash(doc.text);self.db.execute("insert into exact values(?,?)",(exact,doc.document_id))
    self.db.executemany("insert or ignore into para values(?,?)",((x,doc.document_id) for x in paragraph_hashes(doc.text)))
    fp=simhash64(doc.text);stored=fp if fp<(1<<63) else fp-(1<<64)
    self.db.executemany("insert into band values(?,?,?,?)",((b,(fp>>(16*b))&65535,doc.document_id,stored) for b in range(4)))

def init(out,state_path,total):
 if state_path.exists():return json.loads(state_path.read_text())
 out.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(out).free<RESERVE+total*5:raise RuntimeError("insufficient disk safety reserve")
 shutil.copy2(OLD/"dedup.sqlite",out/"dedup.sqlite")
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 state={"schema":"data-d-v3-builder-state-v1","corpus_identity":CORPUS_V3_ID,"builder_git_commit":commit,"target":total,"source":"fineweb_edu","source_index":0,"raw_position":0,"source_tokens":{s:0 for s in SOURCES},"validation_tokens":{s:0 for s in SOURCES},"children":[],"validation_children":[],"next_shard":{s:0 for s in SOURCES},"stats":{s:Counter() for s in SOURCES},"status":"BUILDING"};atomic(state_path,state);return state

def write_block(out,state,index,source,docs,encoded,split):
 number=state["next_shard"][source];suffix="" if split=="train" else "-validation";name=f"{source}{suffix}-{number:04d}.int32"
 partial=out/(name+".partial");final=out/name
 values=np.concatenate([np.asarray(x,dtype="<i4") for x in encoded]);values.tofile(partial)
 with partial.open("rb") as f:os.fsync(f.fileno())
 os.replace(partial,final);bounds=[0]
 for x in encoded:bounds.append(bounds[-1]+len(x))
 repo,config,revision=REVISIONS[source];tokhash=sha256_file(TOKENIZER)
 manifest={"schema_version":SCHEMA_VERSION,"corpus_identity":CORPUS_V3_ID,"source":source,"split":split,"upstream_uri":repo,"upstream_revision":revision,"upstream_configuration":config,"acquisition_timestamp":datetime.now(timezone.utc).isoformat(),"transformation_code_commit":state["builder_git_commit"],"transformation_config_sha256":sha256_bytes(canonical_json({"seed":SEED,"mix":[.5,.25,.25]})),"tokenizer_sha256":tokhash,"token_count":len(values),"document_count":len(docs),"raw_byte_count":sum(len(d.text.encode()) for d in docs),"sha256":sha256_file(final),"source_license_label":LICENSES[source],"document_boundary_offsets":bounds,"stable_document_ids":[d.document_id for d in docs],"document_upstream_metadata":[dict(d.upstream) for d in docs],"validation_assignment":[split]*len(docs),"validation_assignment_method":VALIDATION_ASSIGNMENT_VERSION,"dedup_decision_metadata":{"version":DEDUP_VERSION,"decision":"retained","global_against":["DATA-C","DATA-D-BROAD-v2r1","DATA-D-BROAD-v3"]},"decontamination_metadata":{"version":DECONTAMINATION_VERSION,"decision":"clean"},"path":name}
 mp=out/name.replace(".int32",".manifest.json");mp.write_bytes(canonical_json(manifest));entry={"manifest_path":mp.name,"manifest_sha256":sha256_file(mp)}
 index.add_many(docs);key="children" if split=="train" else "validation_children";state[key].append(entry);state["next_shard"][source]+=1;state[("source_tokens" if split=="train" else "validation_tokens")][source]+=len(values);return entry

def build(out,total,hard_deadline=None,block_tokens=8_000_000,reserve=RESERVE,interrupt_after=None):
 global RESERVE;RESERVE=reserve;state_path=out/"builder-state.json";state=init(out,state_path,total);tok=load_tokenizer(TOKENIZER)
 refs=reference_documents();common,common_meta=common_validation_references(tok);refs["common_validation"]=common;registry=ContaminationRegistry(refs);index=Index(out/"dedup.sqlite")
 targets={"fineweb_edu":total//2,"wikipedia":total//4,"fineweb":total-total//2-total//4};audit=(out/"decisions.jsonl").open("a")
 for si in range(state["source_index"],len(SOURCES)):
  source=SOURCES[si];state["source"]=source;start=state["raw_position"] if si==state["source_index"] else 0;train_docs=[];train_ids=[];val_docs=[];val_ids=[]
  pending_exact={};pending_para={};pending_bands={}
  for pos,doc in enumerate(rows(source)):
   if pos<start:continue
   state["raw_position"]=pos+1
   if hard_deadline and time.time()>hard_deadline-300:atomic(state_path,state);raise TimeoutError("hard deadline safety stop")
   if shutil.disk_usage(out).free<reserve:atomic(state_path,state);raise RuntimeError("insufficient disk safety reserve")
   if doc is None:continue
   hits=registry.matches(doc)
   if hits:state["stats"][source]["decontaminated"]=state["stats"][source].get("decontaminated",0)+1;audit.write(json.dumps({"id":doc.document_id,"decision":"quarantine","hits":hits})+"\n");continue
   rule,winner=index.check(doc)
   if not rule:
    dh=document_hash(doc.text)
    if dh in pending_exact:rule,winner="exact_document",pending_exact[dh]
    else:
     hit=next((("exact_paragraph",pending_para[x]) for x in paragraph_hashes(doc.text) if x in pending_para),None)
     if hit:rule,winner=hit
     else:
      fp=simhash64(doc.text);candidates={ident:value for band in range(4) for ident,value in pending_bands.get((band,(fp>>(16*band))&65535),())};close=sorted((hamming(fp,value),ident) for ident,value in candidates.items())
      if close and close[0][0]<=3:rule,winner="near_duplicate",close[0][1]
   if rule:state["stats"][source][rule]=state["stats"][source].get(rule,0)+1;audit.write(json.dumps({"id":doc.document_id,"decision":"duplicate","reason":rule,"winner":winner})+"\n");continue
   ids=tok.encode(doc.text+"\n");is_val=validation_assignment(doc.document_id)=="validation"
   if is_val and state["validation_tokens"][source]<max(250_000,total//200):val_docs.append(doc);val_ids.append(ids)
   elif not is_val and state["source_tokens"][source]<targets[source]:train_docs.append(doc);train_ids.append(ids)
   else:continue
   dh=document_hash(doc.text);pending_exact[dh]=doc.document_id
   for value in paragraph_hashes(doc.text):pending_para[value]=doc.document_id
   fp=simhash64(doc.text)
   for band in range(4):pending_bands.setdefault((band,(fp>>(16*band))&65535),[]).append((doc.document_id,fp))
   pending=sum(map(len,train_ids))+sum(map(len,val_ids))
   if pending>=block_tokens:
    if train_docs:write_block(out,state,index,source,train_docs,train_ids,"train");train_docs=[];train_ids=[]
    if val_docs:write_block(out,state,index,source,val_docs,val_ids,"validation");val_docs=[];val_ids=[]
    pending_exact={};pending_para={};pending_bands={}
    index.db.commit();audit.flush();os.fsync(audit.fileno());atomic(state_path,state)
    if interrupt_after and len(state["children"])+len(state["validation_children"])>=interrupt_after:raise InterruptedError("fixture interruption")
   if state["source_tokens"][source]>=targets[source] and state["validation_tokens"][source]>=max(250_000,total//200):break
  if train_docs:write_block(out,state,index,source,train_docs,train_ids,"train")
  if val_docs:write_block(out,state,index,source,val_docs,val_ids,"validation")
  if state["source_tokens"][source]<targets[source]:raise RuntimeError(f"{source} exhausted")
  state["source_index"]=si+1;state["raw_position"]=0;atomic(state_path,state)
 audit.close();top={"schema_version":"data-d-corpus-v1","corpus_identity":CORPUS_V3_ID,"prepared":True,"certified":False,"mixture_definition":{"fineweb_edu":.5,"wikipedia":.25,"fineweb":.25},"tokenizer_sha256":sha256_file(TOKENIZER),"dedup_version":DEDUP_VERSION,"decontamination_version":DECONTAMINATION_VERSION,"total_unique_tokens":sum(state["source_tokens"].values()),"total_documents":sum(json.loads((out/x["manifest_path"]).read_text())["document_count"] for x in state["children"]),"canonical_shard_ordering":[x["manifest_path"] for x in state["children"]],"shards":state["children"],"validation_shards":state["validation_children"],"source_stats":state["stats"],"builder_git_commit":state["builder_git_commit"],"common_validation_registry":common_meta,"benchmark_registry_sizes":{k:len(v) for k,v in refs.items() if k!="common_validation"},"overlap_registry":{"base_manifest_sha256":sha256_file(OLD/"manifest.json"),"scope":["DATA-C","DATA-D-BROAD-v2r1"]}}
 (out/"manifest.json").write_bytes(canonical_json(top));state["status"]="COMPLETE";atomic(state_path,state);return top
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUT);p.add_argument("--total-tokens",type=int,default=1_000_000_000);p.add_argument("--hard-deadline-epoch",type=float);a=p.parse_args();print(json.dumps(build(a.output,a.total_tokens,a.hard_deadline_epoch),default=dict))
if __name__=="__main__":main()
