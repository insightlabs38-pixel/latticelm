from __future__ import annotations
import fcntl,hashlib,json,os,shutil
from pathlib import Path
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_json(path,value,backup=None,immutable=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 encoded=(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n").encode()
 if immutable and path.exists():
  if path.read_bytes()!=encoded:raise RuntimeError(f"immutable artifact conflict: {path}")
  return
 tmp=path.with_name("."+path.name+".tmp")
 with tmp.open("wb") as f:f.write(encoded);f.flush();os.fsync(f.fileno())
 if backup and path.exists():shutil.copy2(path,backup)
 os.replace(tmp,path)
def write_sha_sidecar(path):
 path=Path(path);side=path.with_suffix(path.suffix+".sha256");side.write_text(sha256(path)+"\n");return side
class Lock:
 def __init__(self,path):self.path=Path(path);self.file=None
 def __enter__(self):
  self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open("a+");fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB);self.file.seek(0);self.file.truncate();self.file.write(str(os.getpid()));self.file.flush();return self
 def __exit__(self,*_):fcntl.flock(self.file,fcntl.LOCK_UN);self.file.close()
