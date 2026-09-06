"""Build deterministic acquired-source DATA-D-BROAD-v2 fixtures and pilots."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Iterable

import numpy as np
from datasets import load_dataset

from latticelm.data_d import (CORPUS_V2R1_ID, DEDUP_VERSION, DECONTAMINATION_VERSION,
    SCHEMA_VERSION, ContaminationRegistry, Document, canonical_json, document_hash,
    hamming, normalize_text, normalized_words, paragraph_hashes, sha256_bytes,
    sha256_file, simhash64, validation_assignment, VALIDATION_ASSIGNMENT_VERSION)
from latticelm.tokenizer import load_tokenizer
from prepare_phase7a_data import fingerprints, reference_documents, screen

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts"; OLD=ART/"data/phase7a"
TOKENIZER=ART/"tokenizers/babylm_2026_4k.json"; SEED=314159
REVISIONS={
 "fineweb_edu":("HuggingFaceFW/fineweb-edu","sample-10BT","87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
 "wikipedia":("wikimedia/wikipedia","20231101.en","b04c8d1ceb2f5cd4588862100d08de323dccfbaa"),
 "fineweb":("HuggingFaceFW/fineweb","CC-MAIN-2013-20","9bb295ddab0e05d785b879661af7260fed5140fc")}
LICENSES={"fineweb_edu":"ODC-By 1.0 dataset; underlying page terms retained by upstream",
          "wikipedia":"CC BY-SA 4.0 and GFDL; see Wikimedia dump terms",
          "fineweb":"ODC-By 1.0 dataset; underlying page terms retained by upstream"}


class Index:
 def __init__(self,path:Path):
  self.db=sqlite3.connect(path); self.db.execute("pragma journal_mode=WAL"); self.db.execute("pragma synchronous=FULL")
  self.db.executescript("create table if not exists exact(h text primary key,id text); create table if not exists para(h text primary key,id text); create table if not exists band(b integer,k integer,id text,s integer); create index if not exists band_i on band(b,k)")
 def check_add(self,doc:Document,add=True):
  exact=document_hash(doc.text); row=self.db.execute("select id from exact where h=?",(exact,)).fetchone()
  if row:return "exact_document",row[0]
  for value in paragraph_hashes(doc.text):
   row=self.db.execute("select id from para where h=?",(value,)).fetchone()
   if row:return "exact_paragraph",row[0]
  fingerprint=simhash64(doc.text)
  candidates={}
  for band in range(4):
   key=(fingerprint>>(16*band))&65535
   for prior_id,prior in self.db.execute("select id,s from band where b=? and k=?",(band,key)): candidates[prior_id]=prior & ((1<<64)-1)
  close=sorted((hamming(fingerprint,value),key) for key,value in candidates.items())
  if close and close[0][0]<=3:return "near_duplicate",close[0][1]
  if add:
   self.db.execute("insert into exact values(?,?)",(exact,doc.document_id))
   for value in paragraph_hashes(doc.text): self.db.execute("insert or ignore into para values(?,?)",(value,doc.document_id))
   stored=fingerprint if fingerprint < (1<<63) else fingerprint-(1<<64)
   for band in range(4): self.db.execute("insert into band values(?,?,?,?)",(band,(fingerprint>>(16*band))&65535,doc.document_id,stored))
  return None,None
 def commit(self):self.db.commit()


def seed_babylm(index:Index):
 count=0; exact_rows=[]; band_rows=[]
 for path in sorted((OLD/"babylm_strict_100m").glob("*.train.txt")):
  nonempty=sum(bool(x.strip()) for x in path.open(encoding="utf-8",errors="replace")); limit=nonempty-max(1,nonempty//100); seen=0
  with path.open(encoding="utf-8",errors="replace") as handle:
   for line_no,raw in enumerate(handle):
    text=normalize_text(raw)
    if not text:continue
    if seen>=limit:break
    seen+=1
    words=normalized_words(text)
    if len(words) < 8: continue
    doc_id=f"data-c:babylm:{path.name}:{line_no}"; value=document_hash(text)
    exact_rows.append((value,doc_id)); count+=1
    if len(words)>=50:
     fingerprint=simhash64(text);stored=fingerprint if fingerprint<(1<<63) else fingerprint-(1<<64)
     band_rows.extend((band,(fingerprint>>(16*band))&65535,doc_id,stored) for band in range(4))
    if len(exact_rows)>=10000:
     index.db.executemany("insert or ignore into exact values(?,?)",exact_rows)
     index.db.executemany("insert or ignore into para values(?,?)",exact_rows)
     if band_rows:index.db.executemany("insert into band values(?,?,?,?)",band_rows)
     exact_rows=[];band_rows=[]
 if exact_rows:
  index.db.executemany("insert or ignore into exact values(?,?)",exact_rows);index.db.executemany("insert or ignore into para values(?,?)",exact_rows)
  if band_rows:index.db.executemany("insert into band values(?,?,?,?)",band_rows)
 index.commit(); return count


def old_fineweb(index:Index,tokenizer,refs):
 repo,config,revision=REVISIONS["fineweb_edu"]
 stream=load_dataset(repo,config,split="train",streaming=True,revision=revision).shuffle(seed=1337,buffer_size=1000)
 grams,exact=fingerprints(refs);tokens=0; ids=[]; scanned=0
 for row in stream:
  scanned+=1; text=row["text"].strip()
  if len(normalized_words(text))<50 or float(row.get("language_score",1))<.9:continue
  doc=Document("data_c",str(row["id"]),text,row,"upstream")
  if screen(text,grams,exact)[0]:continue
  if int.from_bytes(hashlib.sha256(("1337"+str(row["id"])).encode()).digest()[:8],"big")%20==0:continue
  ids.append(str(row["id"])); tokens+=len(tokenizer.encode(text+"\n")); index.check_add(doc)
  if tokens>=50_000_000:break
 index.commit()
 expected=json.loads((ART/"phase7e_fineweb_expansion_manifest.json").read_text())
 observed=sha256_bytes("\n".join(ids).encode())
 if tokens!=expected["tokens"] or len(ids)!=expected["documents"] or observed!=expected["document_ids_sha256"]:
  raise RuntimeError(f"DATA-C FineWeb replay mismatch: tokens={tokens} docs={len(ids)} ids={observed}")
 return set(ids),scanned


def rows(source:str)->Iterable[Document]:
 repo,config,revision=REVISIONS[source]
 stream=load_dataset(repo,config,split="train",streaming=True,revision=revision)
 if source!="wikipedia":stream=stream.shuffle(seed=SEED+({"fineweb_edu":1,"fineweb":3}[source]),buffer_size=10000)
 for row in stream:
  text=normalize_text(row.get("text", "")); doc_id=str(row.get("id"))
  if len(normalized_words(text))<50:continue
  if source!="wikipedia" and float(row.get("language_score",1))<.95:continue
  if source=="fineweb_edu" and float(row.get("int_score",0))<3:continue
  # Frozen, benchmark-independent quality/procedural filter for general FineWeb.
  if source=="fineweb":
   words=normalized_words(text); procedural=sum(x in words for x in ("how","step","guide","explain","because","use","method"))
   if len(words)<100 or (procedural==0 and len(words)<300):continue
  upstream={k:row.get(k) for k in ("id","url","dump","date","title","file_path","language_score","int_score") if k in row}
  yield Document(source,f"{source}:{doc_id}",text,upstream,LICENSES[source])


class Writer:
 def __init__(self,base:Path,source:str,tokenizer_hash:str,config_hash:str,commit:str,role="train",shard_target=16_000_000):
  self.base=base;self.source=source;self.tokenizer_hash=tokenizer_hash;self.config_hash=config_hash;self.commit=commit;self.shard_target=shard_target
  self.role=role
  self.number=0;self.ids=[];self.bounds=[0];self.assign=[];self.metadata=[];self.tokens=0;self.docs=0;self.raw=0;self.children=[];self.handle=None;self.path=None
 def _open(self):
  suffix="" if self.role=="train" else "-validation"
  self.path=self.base/f"{self.source}{suffix}-{self.number:03d}.int32";self.handle=self.path.open("xb")
 def add(self,doc:Document,ids:list[int]):
  if self.handle is None:self._open()
  if self.tokens and self.tokens+len(ids)>self.shard_target:self.close();self.number+=1;self._open()
  np.asarray(ids,dtype="<i4").tofile(self.handle);self.tokens+=len(ids);self.docs+=1;self.raw+=len(doc.text.encode());self.ids.append(doc.document_id);self.bounds.append(self.tokens);self.assign.append(self.role);self.metadata.append({"upstream":dict(doc.upstream),"normalized_text_sha256":document_hash(doc.text),"paragraph_sha256":paragraph_hashes(doc.text),"simhash64":f"{simhash64(doc.text):016x}"})
 def close(self):
  if self.handle is None:return
  self.handle.flush();os.fsync(self.handle.fileno());self.handle.close()
  repo,config,revision=REVISIONS[self.source]
  manifest={"schema_version":SCHEMA_VERSION,"corpus_identity":CORPUS_V2R1_ID,"source":self.source,"split":self.role,"upstream_uri":repo,"upstream_revision":revision,
   "upstream_configuration":config,"acquisition_timestamp":datetime.now(timezone.utc).isoformat(),"transformation_code_commit":self.commit,
   "transformation_config_sha256":self.config_hash,"tokenizer_sha256":self.tokenizer_hash,"token_count":self.tokens,"document_count":self.docs,
   "raw_byte_count":self.raw,"sha256":sha256_file(self.path),"source_license_label":LICENSES[self.source],"document_boundary_offsets":self.bounds,
   "stable_document_ids":self.ids,"document_upstream_metadata":self.metadata,"validation_assignment":self.assign,"validation_assignment_method":VALIDATION_ASSIGNMENT_VERSION,
   "dedup_decision_metadata":{"version":DEDUP_VERSION,"decision":"retained"},"decontamination_metadata":{"version":DECONTAMINATION_VERSION,"decision":"clean"},"path":self.path.name}
  mp=self.path.with_suffix(".manifest.json");mp.write_bytes(canonical_json(manifest));self.children.append({"manifest_path":mp.name,"manifest_sha256":sha256_file(mp)})
  self.ids=[];self.bounds=[0];self.assign=[];self.metadata=[];self.tokens=0;self.docs=0;self.raw=0;self.handle=None


def common_validation_references(tokenizer):
 manifest=json.loads((ART/"common_validation_manifest.json").read_text()); token_path=ROOT/manifest["common_token_file"]["path"]
 if sha256_file(token_path)!=manifest["common_token_file"]["sha256"]:raise RuntimeError("frozen common-validation hash mismatch")
 values=np.memmap(token_path,mode="r",dtype="<i4")
 # Cover every frozen token while keeping matching units bounded and deterministic.
 rows=[tokenizer.decode([int(x) for x in values[start:start+256]]) for start in range(0,len(values),192)]
 return rows,{"manifest_sha256":sha256_file(ART/"common_validation_manifest.json"),"token_sha256":sha256_file(token_path),"tokens":len(values),"registry_documents":len(rows)}


def audit_row(doc,decision,reason=None,winner=None,hits=None):
 return {"document_id":doc.document_id,"source":doc.source,"upstream":dict(doc.upstream),"license":doc.license_label,
  "normalized_text_sha256":document_hash(doc.text),"normalized_utf8_bytes":len(doc.text.encode()),"normalized_words":len(normalized_words(doc.text)),
  "paragraph_sha256":paragraph_hashes(doc.text),"simhash64":f"{simhash64(doc.text):016x}","cluster_id":winner or doc.document_id,
  "decision":decision,"reason":reason or decision,"contamination_hits":hits or []}


def main():
 p=argparse.ArgumentParser();p.add_argument("--total-tokens",type=int,required=True);p.add_argument("--name",required=True);p.add_argument("--reuse-babylm-index",action="store_true");a=p.parse_args()
 base=ART/"data/phase7f_v2r1"/a.name
 if base.exists() and any(base.iterdir()):
  allowed={"dedup.sqlite","dedup.sqlite-wal","dedup.sqlite-shm"}
  if not a.reuse_babylm_index or any(x.name not in allowed for x in base.iterdir()):raise RuntimeError(f"refusing to overwrite {base}")
 base.mkdir(parents=True,exist_ok=True);tok=load_tokenizer(TOKENIZER);tokenizer_hash=sha256_file(TOKENIZER)
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():raise RuntimeError("canonical/fixture generation requires a clean committed source tree")
 benchmark_refs=reference_documents();refs=dict(benchmark_refs);common_rows,common_meta=common_validation_references(tok);refs["common_validation"]=common_rows;registry=ContaminationRegistry(refs)
 index=Index(base/"dedup.sqlite");old_ids=set();old_stats={}
 if a.reuse_babylm_index:
  old_stats["babylm_documents_indexed"]=index.db.execute("select count(*) from exact").fetchone()[0]
 else: old_stats["babylm_documents_indexed"]=seed_babylm(index)
 old_ids,scanned=old_fineweb(index,tok,benchmark_refs);old_stats["fineweb_documents_indexed"]=len(old_ids);old_stats["fineweb_rows_scanned"]=scanned
 config={"corpus_identity":CORPUS_V2R1_ID,"targets":{"fineweb_edu":.5,"wikipedia":.25,"fineweb":.25},"seed":SEED,"minimum_words":50,"web_language_score":.95,"fineweb_edu_int_score":3,"validation_assignment":VALIDATION_ASSIGNMENT_VERSION}
 config_hash=sha256_bytes(canonical_json(config));targets={"fineweb_edu":a.total_tokens//2,"wikipedia":a.total_tokens//4,"fineweb":a.total_tokens-a.total_tokens//2-a.total_tokens//4}
 stats={};all_children=[];validation_children=[];decon=[];dedup=[]
 raw_cache=(base/"normalized-source-cache.jsonl").open("x",encoding="utf-8");audit=(base/"document-audit.jsonl").open("x",encoding="utf-8")
 for source in ("fineweb_edu","wikipedia","fineweb"):
  writer=Writer(base,source,tokenizer_hash,config_hash,commit,shard_target=min(16_000_000,max(256_000,targets[source])))
  val_writer=Writer(base,source,tokenizer_hash,config_hash,commit,role="validation",shard_target=max(64_000,min(1_000_000,targets[source]//50)))
  train=val=raw_docs=0
  for doc in rows(source):
   raw_docs+=1
   raw_cache.write(json.dumps({"document_id":doc.document_id,"source":source,"upstream":dict(doc.upstream),"normalized_text":doc.text},sort_keys=True)+"\n")
   if source=="fineweb_edu" and doc.document_id.split(":",1)[1] in old_ids:
    row=audit_row(doc,"rejected","data_c_id","data-c:fineweb-edu:"+doc.document_id.split(":",1)[1]);dedup.append(row);audit.write(json.dumps(row,sort_keys=True)+"\n");continue
   hits=registry.matches(doc)
   if hits:
    row=audit_row(doc,"rejected","contamination",hits=hits);decon.extend(hits);audit.write(json.dumps(row,sort_keys=True)+"\n");continue
   split=validation_assignment(doc.document_id)
   if split=="train" and train>=targets[source]:
    audit.write(json.dumps(audit_row(doc,"not_selected","target_already_full"),sort_keys=True)+"\n");continue
   rule,winner=index.check_add(doc)
   if rule:
    row=audit_row(doc,"rejected",rule,winner);dedup.append(row);audit.write(json.dumps(row,sort_keys=True)+"\n");continue
   ids=tok.encode(doc.text+"\n")
   if split=="validation":val_writer.add(doc,ids);val+=len(ids)
   elif train<targets[source]:writer.add(doc,ids);train+=len(ids)
   row=audit_row(doc,"retained",split);audit.write(json.dumps(row,sort_keys=True)+"\n")
   if train>=targets[source] and val>=min(250_000,max(32_000,targets[source]//50)):break
   if raw_docs%1000==0:index.commit()
  writer.close();val_writer.close();index.commit();all_children.extend(writer.children);validation_children.extend(val_writer.children)
  stats[source]={"target_tokens":targets[source],"retained_training_tokens":train,"validation_tokens":val,"raw_documents_examined":raw_docs}
  if train<targets[source]:raise RuntimeError(f"{source} exhausted at {train}")
 raw_cache.flush();os.fsync(raw_cache.fileno());raw_cache.close();audit.flush();os.fsync(audit.fileno());audit.close()
 top={"schema_version":"data-d-corpus-v1","corpus_identity":CORPUS_V2R1_ID,"prepared":True,"certified":False,"mixture_definition":{"fineweb_edu":.5,"wikipedia":.25,"fineweb":.25},
  "tokenizer_sha256":tokenizer_hash,"dedup_version":DEDUP_VERSION,"decontamination_version":DECONTAMINATION_VERSION,"total_unique_tokens":sum(x["retained_training_tokens"] for x in stats.values()),
  "total_documents":sum(json.loads((base/x["manifest_path"]).read_text())["document_count"] for x in all_children),"canonical_shard_ordering":[x["manifest_path"] for x in all_children],"shards":all_children,
  "source_stats":stats,"data_c_index_stats":old_stats,"transformation_config":config,"builder_git_commit":commit,"common_validation_registry":common_meta,
  "benchmark_registry_sizes":{key:len(value) for key,value in refs.items() if key!="common_validation"},"validation_shards":validation_children,
  "audit_files":{"normalized_source_cache":{"path":"normalized-source-cache.jsonl","sha256":sha256_file(base/"normalized-source-cache.jsonl")},"document_audit":{"path":"document-audit.jsonl","sha256":sha256_file(base/"document-audit.jsonl")}},
  "dedup_removals":Counter(x["reason"] for x in dedup),"decontamination_removals":Counter(x["benchmark"] for x in decon)}
 (base/"manifest.json").write_bytes(canonical_json(top));(base/"dedup-decisions.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in dedup));(base/"decontamination-decisions.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in decon))
 print(json.dumps({"manifest":str(base/"manifest.json"),"sha256":sha256_file(base/"manifest.json"),"stats":stats,"old":old_stats},indent=2))
if __name__=="__main__":main()
