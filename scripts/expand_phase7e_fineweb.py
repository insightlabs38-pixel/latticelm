"""Atomically expand DATA-C FineWeb-Edu while proving the old stream is a prefix."""
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from datasets import load_dataset
from latticelm.tokenizer import load_tokenizer
from prepare_phase7a_data import FINEWEB_CONFIG, FINEWEB_REPO, FINEWEB_REVISION, SEED, fingerprints, reference_documents, screen, normalized_words

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"artifacts/data/phase7a"; ART=ROOT/"artifacts"
def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(); p.add_argument("--tokens",type=int,default=50_000_000); a=p.parse_args()
 old=DATA/"finewebedu_train.int32"; old_hash=digest(old); old_tokens=old.stat().st_size//4
 tok=load_tokenizer(ART/"tokenizers/babylm_2026_4k.json"); refs=reference_documents(); grams,exact=fingerprints(refs)
 stream=load_dataset(FINEWEB_REPO,FINEWEB_CONFIG,split="train",streaming=True,revision=FINEWEB_REVISION).shuffle(seed=SEED,buffer_size=1000)
 tmp=DATA/f".finewebedu_train.{os.getpid()}.int32"; accepted=scanned=documents=0; ids=[]
 try:
  with tmp.open("wb") as out:
   for row in stream:
    scanned+=1; text=row["text"].strip()
    if len(normalized_words(text))<50 or float(row.get("language_score",1))<.9: continue
    remove,_,_=screen(text,grams,exact)
    if remove: continue
    token_ids=tok.encode(text+"\n")
    val=int.from_bytes(hashlib.sha256((str(SEED)+row["id"]).encode()).digest()[:8],"big")%20==0
    if val: continue
    np.asarray(token_ids,dtype="<i4").tofile(out); accepted+=len(token_ids); documents+=1; ids.append(row["id"])
    if accepted>=a.tokens: break
   out.flush(); os.fsync(out.fileno())
  if accepted<a.tokens: raise RuntimeError(f"source exhausted at {accepted} tokens")
  with old.open("rb") as left,tmp.open("rb") as right:
   while True:
    block=left.read(1<<20)
    if not block: break
    if right.read(len(block))!=block: raise RuntimeError("expanded stream does not preserve canonical DATA-C prefix")
  os.replace(tmp,old)
  manifest={"schema_version":1,"source":"FineWeb-Edu","repo":FINEWEB_REPO,"config":FINEWEB_CONFIG,
   "revision":FINEWEB_REVISION,"seed":SEED,"tokens":accepted,"documents":documents,"scanned":scanned,
   "sha256":digest(old),"prior_tokens":old_tokens,"prior_sha256":old_hash,"old_stream_exact_prefix":True,
   "document_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),"created_at":datetime.now(timezone.utc).isoformat()}
  (ART/"phase7e_fineweb_expansion_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
  print(json.dumps(manifest))
 finally:
  if tmp.exists(): tmp.unlink()
if __name__=="__main__": main()
