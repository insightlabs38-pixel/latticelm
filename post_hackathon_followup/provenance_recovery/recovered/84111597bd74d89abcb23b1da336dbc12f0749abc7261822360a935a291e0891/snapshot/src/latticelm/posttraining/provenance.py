"""Restart-safe, append-only provenance shards.

Only a fully written line is visible. A retry of an existing cursor must match
the original bytes. Closed shards carry a SHA sidecar.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
_CACHE={}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def append_record(root,index,value,shard_size=1024):
 if index<0 or shard_size<1:raise ValueError("invalid provenance cursor")
 root=Path(root);root.mkdir(parents=True,exist_ok=True);start=index//shard_size*shard_size
 path=root/f"{start:012d}-{start+shard_size-1:012d}.jsonl"
 row=(json.dumps({"cursor":index,"record":value},sort_keys=True,separators=(",",":"),default=str)+"\n").encode()
 key=str(path.resolve());state=_CACHE.get(key)
 if state is None:
  data=path.read_bytes() if path.exists() else b"";lines=data.splitlines(keepends=True)
  if lines and not lines[-1].endswith(b"\n"):
   data=b"".join(lines[:-1]);path.write_bytes(data);lines=lines[:-1]
  side=path.with_suffix(".sha256")
  if side.exists() and side.read_text().strip()!=hashlib.sha256(data).hexdigest():
   prefix=b"".join(lines[:-1])
   if not lines or hashlib.sha256(prefix).hexdigest()!=side.read_text().strip():raise RuntimeError("provenance shard SHA mismatch")
   side.write_text(hashlib.sha256(data).hexdigest()+"\n")
  records={json.loads(line)["cursor"]:line for line in lines};h=hashlib.sha256();h.update(data);state={"records":records,"hash":h,"last":max(records,default=start-1)}
  if len(_CACHE)>2:_CACHE.clear()
  _CACHE[key]=state
 if index in state["records"]:
  if state["records"][index]!=row:raise RuntimeError("immutable provenance conflict")
  return str(path)
 if index<state["last"]:raise RuntimeError("provenance cursor moved backwards")
 with path.open("ab") as f:f.write(row);f.flush();os.fsync(f.fileno())
 state["hash"].update(row);state["records"][index]=row;state["last"]=index
 side=path.with_suffix(".sha256");tmp=side.with_suffix(".sha256.tmp")
 with tmp.open("w") as f:f.write(state["hash"].hexdigest()+"\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,side)
 return str(path)
