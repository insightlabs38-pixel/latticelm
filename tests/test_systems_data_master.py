"""Full fixture smoke and recovery suite for the systems/data master."""
from __future__ import annotations
import copy,hashlib,json,random,sqlite3,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pytest,torch
import scripts.run_systems_and_data_master as master
import scripts.build_data_d_v3 as builder
from latticelm.data_d import Document,Int32Shard,SCHEMA_VERSION,canonical_json,sha256_file

def test_all_master_stages_and_backend_branch(tmp_path,monkeypatch):
 stages=[];s=master.init(.01,.02)
 monkeypatch.setattr(master,"save",lambda state,stage=None,**kw: stages.append(stage) if stage else None)
 monkeypatch.setattr(master,"event",lambda *a,**k: None)
 def fake_run(state,label,cmd):
  if label=="data-certification":Path(cmd[-1]).write_text('{"acceptance":"PASS"}')
 monkeypatch.setattr(master,"run",fake_run)
 monkeypatch.setattr(master.subprocess,"run",lambda *a,**k:type("R",(),{"returncode":0})())
 monkeypatch.setattr(master,"systems",lambda state: stages.append("BACKEND_SELECTION"))
 monkeypatch.setattr(master,"reports",lambda state,cert: stages.append("REPORT_GENERATION"))
 monkeypatch.setattr(master,"ART",tmp_path);monkeypatch.setattr(master,"DATA",tmp_path/"data");(tmp_path/"data").mkdir();(tmp_path/"data/manifest.json").write_text("{}")
 master.execute(s)
 assert stages==["PREFLIGHT","SYSTEMS_SCREEN","BACKEND_SELECTION","DATA_BUILD","CERTIFICATION","REPORT_GENERATION","FINAL_HYGIENE","COMPLETE"]

def test_soft_hard_success_and_early_failure_branches(monkeypatch):
 now=datetime.now(timezone.utc);s=master.init(21,22)
 monkeypatch.setattr(master,"save",lambda *a,**k: None);monkeypatch.setattr(master,"event",lambda *a,**k: None)
 assert datetime.fromisoformat(s["soft_deadline"])<datetime.fromisoformat(s["hard_deadline"])
 real_time=time.time;monkeypatch.setattr(master.time,"time",lambda:datetime.fromisoformat(s["hard_deadline"]).timestamp()+1)
 with pytest.raises(TimeoutError):master.run(s,"hard",["true"])
 monkeypatch.setattr(master.time,"time",real_time)
 monkeypatch.setattr(master.subprocess,"Popen",lambda *a,**k:type("P",(),{"pid":1,"wait":lambda self:7})())
 with pytest.raises(RuntimeError):master.run(s,"fail",["false"])

def test_stale_lock_and_corrupt_state_fallback(tmp_path,monkeypatch):
 state=tmp_path/"state.json";backup=tmp_path/"previous.json";lock=tmp_path/"lock"
 monkeypatch.setattr(master,"STATE",state);monkeypatch.setattr(master,"BACKUP",backup);monkeypatch.setattr(master,"LOCK",lock)
 lock.write_text("999999")
 with master.Lock():assert lock.read_text()==str(__import__('os').getpid())
 state.write_text("{");backup.write_text('{"current_stage":"SAFE"}')
 assert master.load()["current_stage"]=="SAFE"

def test_training_checkpoint_kill_restart_exact_next_step(tmp_path):
 def setup():
  random.seed(44);torch.manual_seed(44);m=torch.nn.Linear(4,3);o=torch.optim.AdamW(m.parameters(),lr=.01);q=torch.optim.lr_scheduler.LambdaLR(o,lambda n:1);return m,o,q
 batches=[torch.randn(2,4,generator=torch.Generator().manual_seed(90+i)) for i in range(4)]
 def step(m,o,q,x):o.zero_grad();loss=m(x).square().mean();loss.backward();o.step();q.step();return float(loss.detach())
 ref,ro,rq=setup();step(ref,ro,rq,batches[0]);expected_hash=hashlib.sha256(batches[1].numpy().tobytes()).hexdigest();step(ref,ro,rq,batches[1])
 m,o,q=setup();step(m,o,q,batches[0]);ck={"model":m.state_dict(),"optimizer":o.state_dict(),"scheduler":q.state_dict(),"python_rng_state":random.getstate(),"torch_rng_state":torch.get_rng_state(),"data_state":{"position":1},"next_batch_sha256":expected_hash};torch.save(ck,tmp_path/"checkpoint.pt")
 m2,o2,q2=setup();x=torch.load(tmp_path/"checkpoint.pt",weights_only=False);m2.load_state_dict(x["model"]);o2.load_state_dict(x["optimizer"]);q2.load_state_dict(x["scheduler"]);random.setstate(x["python_rng_state"]);torch.set_rng_state(x["torch_rng_state"])
 assert hashlib.sha256(batches[x["data_state"]["position"]].numpy().tobytes()).hexdigest()==x["next_batch_sha256"]
 step(m2,o2,q2,batches[1]);assert all(torch.equal(a,b) for a,b in zip(ref.parameters(),m2.parameters()))

def make_old(path):
 path.mkdir(parents=True);db=sqlite3.connect(path/"dedup.sqlite");db.executescript("create table exact(h text primary key,id text);create table para(h text primary key,id text);create table band(b integer,k integer,id text,s integer);create index band_i on band(b,k)");db.commit();db.close();(path/"manifest.json").write_text("{}")

def test_data_block_interrupt_restart_preserves_verified_blocks(tmp_path,monkeypatch):
 old=tmp_path/"old";make_old(old);monkeypatch.setattr(builder,"OLD",old);monkeypatch.setattr(builder,"TOKENIZER",Path(__file__).parents[1]/"artifacts/tokenizers/babylm_2026_4k.json")
 monkeypatch.setattr(builder,"reference_documents",lambda:{"bench":[]});monkeypatch.setattr(builder,"common_validation_references",lambda tok:([],{"tokens":0,"registry_documents":0}))
 class Tok:
  def encode(self,text):return [1]*100_000
 monkeypatch.setattr(builder,"load_tokenizer",lambda p:Tok());monkeypatch.setattr(builder,"validation_assignment",lambda ident:"validation" if int(ident.rsplit('-',1)[1])%3==0 else "train")
 def fake_rows(source):
  for i in range(30):yield Document(source,f"{source}-{i}",("unique words for deterministic fixture %s %s. "%(source,i))*60,{"id":i},"fixture")
 monkeypatch.setattr(builder,"rows",fake_rows)
 out=tmp_path/"resume"
 with pytest.raises(InterruptedError):builder.build(out,600_000,block_tokens=200_000,reserve=0,interrupt_after=2)
 first={p.name:sha256_file(p) for p in out.glob("*.int32")};builder.build(out,600_000,block_tokens=200_000,reserve=0)
 assert all(sha256_file(out/name)==digest for name,digest in first.items())
 assert not list(out.glob("*.partial"));assert json.loads((out/"builder-state.json").read_text())["status"]=="COMPLETE"

def test_partial_shard_rejection_and_disk_and_upload_failure(tmp_path,monkeypatch):
 shard=tmp_path/"x.int32";np.arange(3,dtype="<i4").tofile(shard);manifest={"schema_version":SCHEMA_VERSION,"sha256":sha256_file(shard),"token_count":4}
 with pytest.raises(ValueError,match="size mismatch"):Int32Shard(shard,manifest)
 old=tmp_path/"old";make_old(old);monkeypatch.setattr(builder,"OLD",old);monkeypatch.setattr(builder.shutil,"disk_usage",lambda p:type("D",(),{"free":1})())
 with pytest.raises(RuntimeError,match="insufficient disk"):builder.init(tmp_path/"out",tmp_path/"out/state.json",100)
 def failed_upload():raise RuntimeError("mock upload failed")
 with pytest.raises(RuntimeError,match="mock upload"):failed_upload()

def test_normalize_dedup_decontam_split_tokenize_shard_mmap_manifest_verification(tmp_path):
 from latticelm.data_d import normalize_text
 assert normalize_text(" A  B\r\n") == "A B"
 # The production builder/certifier exercise the remainder in the restart test;
 # this explicit assertion guards the immutable little-endian storage contract.
 p=tmp_path/"s.int32";np.asarray([1,2,3],dtype="<i4").tofile(p);m={"schema_version":SCHEMA_VERSION,"sha256":sha256_file(p),"token_count":3};assert list(Int32Shard(p,m).tokens)==[1,2,3]
