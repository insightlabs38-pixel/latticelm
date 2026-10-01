#!/usr/bin/env python3
"""One-shot, restart-safe recovery for rollout certification before RLVR."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; PT=ROOT/'artifacts/posttraining'; STATE=PT/'master_state.json'; CERT=PT/'rollout_backend_certification.json'; LOCK=PT/'watch_rlvr_gate.lock'
MASTER='scripts/run_posttraining_master.py'; WORKER='scripts/train_posttraining_worker.py'
STOPPED=None
GUARDIAN=None

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,data,backup=None):
 path=Path(path);tmp=path.with_name('.'+path.name+'.watchdog.tmp')
 with tmp.open('w') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 if backup and path.exists():
  import shutil;shutil.copy2(path,backup)
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def require_unchanged(path,expected_sha):
 if digest(path)!=expected_sha:raise RuntimeError('master state changed unexpectedly while frozen')
def recovery_state(state,certification=None,failure=None,simpo=None,now=None):
 updated=dict(state)
 if certification is not None:
  updated['rollout_backend']='CERTIFIED';updated['rollout_topology']=certification['selected'];updated['rollout_recertification']={'status':'PASS','reason':'recovered from prior runtime-only certification failure','certification_path':str(CERT),'certification_sha256':digest(CERT),'timestamp':now or datetime.now(timezone.utc).isoformat()}
 else:updated['rollout_recertification']={'status':'FAIL','reason':failure or 'unknown certification failure','timestamp':now or datetime.now(timezone.utc).isoformat(),'certification_log':str(PT/'logs/rollout-recertification-watchdog.log')}
 if simpo is not None:
  updated['watchdog_simpo_outcome']={k:v for k,v in simpo.items() if k!='worker_result'}
  rr=simpo.get('worker_result')
  if rr and simpo.get('status') in ('COMPLETE','SAFE_STOPPED_VALID'):
   wall=max(float(rr.get('wall_seconds',0)),1e-9)
   updated.setdefault('candidates',{})['simpo']={**rr,'candidate_id':'simpo','optimizer':rr.get('optimizer'),'lr':rr.get('lr'),'replay':rr.get('replay'),'status':rr['status'],'method_chain':rr.get('method_chain',[]),'requested_tokens':simpo.get('requested_tokens'),'elapsed_seconds':wall,'observed_logical_tokens_per_second':rr.get('training_tokens',0)/wall}
   if rr['status']=='SAFE_STOPPED_VALID':
    logical=int(rr.get('training_tokens',0));processed=int(rr.get('processed_tokens',0));updates=int(rr.get('updates',0))
    updated['simpo_early_stop']={'reason':'measured wall-clock cost invalidated logical-token scheduling estimate','status':rr['status'],'requested_training_tokens':simpo.get('requested_tokens'),'actual_training_tokens':logical,'actual_processed_tokens':processed,'updates':updates,'wall_seconds':wall,'logical_tokens_per_second':logical/wall,'processed_tokens_per_second':processed/wall,'processed_to_logical_ratio':processed/max(1,logical),'stopped_at':now or datetime.now(timezone.utc).isoformat(),'preserved_as_candidate':True}
 return updated
def proc_rows():
 rows=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   raw=(d/'cmdline').read_bytes().split(b'\0');argv=[x.decode(errors='replace') for x in raw if x];status=(d/'status').read_text();state=next(x.split()[1] for x in status.splitlines() if x.startswith('State:'));ppid=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
   rows.append({'pid':int(d.name),'ppid':ppid,'state':state,'argv':argv,'cmd':' '.join(argv),'cwd':os.readlink(d/'cwd')})
  except (OSError,StopIteration,ValueError):pass
 return rows
def identify():
 s=json.loads(STATE.read_text()); rows=proc_rows()
 masters=[p for p in rows if any(x.endswith(MASTER) for x in p['argv'])]
 if s.get('schema')!='posttraining-master-state-v1':raise RuntimeError('unexpected master state schema')
 if s.get('phase')!='SIMPO_TOURNAMENT':raise RuntimeError(f"unsafe phase: {s.get('phase')}")
 if len(masters)!=1:raise RuntimeError(f'expected exactly one master, found {len(masters)}')
 m=masters[0]; children=[p for p in rows if p['ppid']==m['pid'] and any(x.endswith(WORKER) for x in p['argv']) and '--method' in p['argv'] and p['argv'][p['argv'].index('--method')+1]=='simpo' and '--output' in p['argv'] and Path(p['argv'][p['argv'].index('--output')+1]).resolve()==(PT/'candidates/simpo').resolve()]
 if len(children)!=1:raise RuntimeError(f'expected exactly one active SIMPO child, found {len(children)}')
 c=children[0]
 if int(s.get('child_pid') or -1)!=c['pid'] or s.get('child_label')!='simpo':raise RuntimeError('SIMPO child does not match persisted child identity')
 if Path(m['cwd']).resolve()!=ROOT:raise RuntimeError('master working directory mismatch')
 return s,m,c

def validate_cert(path,base,tokenizer):
 x=json.loads(Path(path).read_text())
 if x.get('schema')!='rollout-backend-certification-v1' or x.get('status')!='PASS':raise RuntimeError('certification schema/status invalid')
 selected=x.get('selected')
 if not isinstance(selected,dict) or not selected.get('online_worker_supported') or selected.get('processes')!=1:raise RuntimeError('no valid online selected topology')
 if selected not in x.get('topologies',[]):raise RuntimeError('selected topology is not in benchmark results')
 if x.get('trajectory_parity',{}).get('status')!='PASS':raise RuntimeError('trajectory parity not PASS')
 if x.get('checkpoint_sha256')!=digest(base) or x.get('tokenizer_sha256')!=digest(tokenizer):raise RuntimeError('certification identity hash mismatch')
 if not x.get('topologies') or any(not r.get('stable') for r in x['topologies']):raise RuntimeError('topology benchmark incomplete')
 for row in x['topologies']:
  for key in ('aggregate_tokens_per_second','wall_seconds'):
   if not math.isfinite(float(row.get(key,0))) or float(row[key])<=0:raise RuntimeError(f'invalid topology throughput: {key}')
 return x

def validate_simpo_result(result_path,checkpoint,child):
 import torch
 rr=json.loads(Path(result_path).read_text())
 if rr.get('schema')!='posttraining-worker-result-v2' or rr.get('method')!='simpo' or rr.get('status') not in ('COMPLETE','SAFE_STOPPED_VALID'):raise RuntimeError('SIMPO result is not a usable terminal worker result')
 if int(rr.get('training_tokens',0))<=0 or int(rr.get('updates',0))<=0:raise RuntimeError('SIMPO result contains no completed training update')
 if not Path(checkpoint).is_file() or digest(checkpoint)!=rr.get('checkpoint_sha256'):raise RuntimeError('SIMPO checkpoint SHA mismatch')
 args=child['argv'];base=Path(args[args.index('--base')+1]);manifest=Path(args[args.index('--manifest')+1]);tokenizer=ROOT/'artifacts/tokenizers/final_corpus_4k.json'
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 expected={'method':'simpo','parent_checkpoint_sha256':digest(base),'tokenizer_sha256':digest(tokenizer),'manifest_sha256':digest(manifest)}
 for key,value in expected.items():
  if ck.get(key)!=value:raise RuntimeError(f'SIMPO checkpoint {key} identity mismatch')
 if int(ck.get('training_tokens',-1))!=int(rr['training_tokens']) or int(ck.get('updates',-1))!=int(rr['updates']) or int(ck.get('processed_tokens',-1))!=int(rr.get('processed_tokens',-2)):raise RuntimeError('SIMPO checkpoint/result counters disagree')
 if not isinstance(ck.get('model'),dict) or not ck['model']:raise RuntimeError('SIMPO checkpoint model state is empty')
 for name,tensor in ck['model'].items():
  if isinstance(tensor,torch.Tensor) and not torch.isfinite(tensor).all().item():raise RuntimeError(f'nonfinite SIMPO tensor: {name}')
 side=Path(checkpoint).with_suffix('.pt.sha256')
 if side.exists() and side.read_text().strip()!=rr['checkpoint_sha256']:raise RuntimeError('SIMPO checkpoint sidecar SHA mismatch')
 return rr

def dry_run():
 s,m,c=identify();base=Path(s['base']['checkpoint']);tok=ROOT/'artifacts/tokenizers/final_corpus_4k.json';cmd=[sys.executable,'scripts/certify_rollout_backend.py','--checkpoint',str(base),'--tokenizer',str(tok),'--output',str(CERT)+'.watchdog.tmp']
 return {'status':'DRY_RUN_PASS','master_pid':m['pid'],'simpo_pid':c['pid'],'state_sha256':digest(STATE),'phase':s['phase'],'result_path':str(PT/'candidates/simpo/result.json'),'checkpoint_path':str(PT/'candidates/simpo/latest.pt'),'base_checkpoint':str(base),'certification_command':cmd,'canonical_certification':str(CERT),'intended_mutation':{'rollout_backend':'CERTIFIED','rollout_topology':'certification.selected','rollout_recertification':'audit record'},'restart':['SIGTERM old master after SIGCONT','launch same scripts/run_posttraining_master.py command'], 'signals_sent':False,'state_mutated':False}

def stopped_cleanup(*_):
 global STOPPED
 if STOPPED:
  try:os.kill(STOPPED,signal.SIGCONT)
  except ProcessLookupError:pass
  STOPPED=None
 if _:
  raise SystemExit(128+int(_[0]) if isinstance(_[0],int) else 1)

def guardian(master_pid):
 """Resume the frozen master if this watchdog's parent dies unexpectedly."""
 import ctypes
 parent=os.getppid();libc=ctypes.CDLL(None,use_errno=True)
 # Linux PR_SET_PDEATHSIG: kernel delivers SIGTERM when watchdog exits.
 if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'prctl(PR_SET_PDEATHSIG)')
 def resume(*_):
  try:
   p=next((x for x in proc_rows() if x['pid']==master_pid and any(z.endswith(MASTER) for z in x['argv'])),None)
   if p and 'T' in p['state']:os.kill(master_pid,signal.SIGCONT)
  except (OSError,StopIteration):pass
  raise SystemExit(0)
 signal.signal(signal.SIGTERM,resume);signal.signal(signal.SIGINT,resume)
 if os.getppid()!=parent:resume()
 while True:time.sleep(30)

def run(a):
 global STOPPED,GUARDIAN
 PT.mkdir(parents=True,exist_ok=True)
 with LOCK.open('a+') as lf:
  fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
  s,m,c=identify();initial_hash=digest(STATE);base=Path(s['base']['checkpoint']);tokenizer=ROOT/'artifacts/tokenizers/final_corpus_4k.json'
  if digest(base)!=s['base']['checkpoint_sha256']:raise RuntimeError('frozen BASE SHA mismatch')
  os.kill(m['pid'],signal.SIGSTOP);STOPPED=m['pid']
  GUARDIAN=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--guardian-master',str(m['pid'])],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
  time.sleep(.5)
  if 'T' not in next(p for p in proc_rows() if p['pid']==m['pid'])['state']:raise RuntimeError('master did not stop')
  child=next((p for p in proc_rows() if p['pid']==c['pid']),None)
  if child is None:raise RuntimeError('SIMPO child unexpectedly exited during freeze')
  while True:
   child=next((p for p in proc_rows() if p['pid']==c['pid']),None)
   if child is None or child['state']=='Z':break
   time.sleep(max(2,min(5,a.poll_seconds)))
  # Child may be a zombie until its paused parent reaps it; inspect persisted result.
  result_path=PT/'candidates/simpo/result.json';checkpoint=PT/'candidates/simpo/latest.pt';simpo={'status':'FAILED_OR_NO_RESULT'}
  if result_path.exists():
   try:
    rr=validate_simpo_result(result_path,checkpoint,c);status=rr['status']
    simpo={'status':status,'result_path':str(result_path),'checkpoint':str(checkpoint),'checkpoint_sha256':rr['checkpoint_sha256'],'worker_result':rr,'requested_tokens':int(c['argv'][c['argv'].index('--tokens')+1])}
   except Exception as e:simpo={'status':'INVALID_RESULT','error':str(e),'result_path':str(result_path)}
  if a.after_stop_hook:subprocess.run(a.after_stop_hook,shell=True,check=True)
  require_unchanged(STATE,initial_hash)
  temp=CERT.with_name('.rollout_backend_certification.watchdog.tmp.json')
  cmd=[sys.executable,'scripts/certify_rollout_backend.py','--checkpoint',str(base),'--tokenizer',str(tokenizer),'--output',str(temp)]
  log=PT/'logs/rollout-recertification-watchdog.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('a') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
  passed=False;cert=None
  if code==0:
   try:
    cert=validate_cert(temp,base,tokenizer)
    with temp.open('rb') as f:os.fsync(f.fileno())
    os.replace(temp,CERT);fd=os.open(CERT.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd);passed=True
   except Exception as e:failure=f'{type(e).__name__}: {e}'
  else:failure=f'certification process exit code {code}'
  require_unchanged(STATE,initial_hash)
  now=datetime.now(timezone.utc).isoformat(); state=recovery_state(json.loads(STATE.read_text()),cert, failure if not passed else None,simpo,now)
  atomic_json(STATE,state,PT/'master_state.watchdog-backup.json')
  # Gracefully release the parent's lock, then relaunch with its canonical argv.
  old_argv=m['argv'];os.kill(m['pid'],signal.SIGCONT);STOPPED=None;os.kill(m['pid'],signal.SIGTERM)
  deadline=time.time()+30
  while time.time()<deadline:
   if not Path(f'/proc/{m["pid"]}').exists():break
   time.sleep(.25)
  if Path(f'/proc/{m["pid"]}').exists():raise RuntimeError('old master did not exit; refusing competing restart')
  new=subprocess.Popen(old_argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(PT/'logs/master-watchdog-restart.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
  restart={'pid':new.pid,'method':'same canonical argv, cwd, and restart-safe state machine'}
  # Confirm new process/lock/state progression; don't wait through all future training.
  deadline=time.time()+3600
  while time.time()<deadline:
   time.sleep(2)
   try:current=json.loads(STATE.read_text())
   except Exception:continue
   if current.get('phase') in ('RLVR_ELIGIBILITY','RLVR_PILOTS','RLVR_EXTENSION','RECOVERY_ANNEAL','INTERPOLATION','ADAPTIVE_RESEARCH','FINAL_CANDIDATE_TOURNAMENT','FINAL_SELECTION','FINAL_EXPORT','FINAL_EVIDENCE_PACKAGE','POSTTRAINING_COMPLETE'):
    if simpo.get('status') in ('COMPLETE','SAFE_STOPPED_VALID'):
     saved=current.get('candidates',{}).get('simpo',{})
     if saved.get('checkpoint_sha256')!=simpo.get('checkpoint_sha256'):raise RuntimeError('restarted master did not load the completed SIMPO candidate')
     running_simpo=[p for p in proc_rows() if p['ppid']==new.pid and any(x.endswith(WORKER) for x in p['argv']) and '--method' in p['argv'] and p['argv'][p['argv'].index('--method')+1]=='simpo']
     if running_simpo:raise RuntimeError('restarted master started SIMPO again despite a terminal candidate')
    break
  else:raise RuntimeError('replacement master did not progress to/through RLVR_ELIGIBILITY')
  report={'status':'RECOVERY_COMPLETE','timestamp':datetime.now(timezone.utc).isoformat(),'old_master_pid':m['pid'],'new_master_pid':new.pid,'simpo_pid':c['pid'],'initial_state_sha256':initial_hash,'initial_phase':s['phase'],'simpo_outcome':{k:v for k,v in simpo.items() if k!='worker_result'},'certification':'PASS' if passed else 'FAIL','selected':cert['selected'] if cert else None,'certification_failure':None if passed else failure,'phase_after_restart':current.get('phase'),'restart':restart,'log':str(log),'certification_path':str(CERT) if passed else None}
  atomic_json(PT/'watch_rlvr_gate_result.json',report);print(json.dumps(report,indent=2))
  return 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--poll-seconds',type=float,default=3);p.add_argument('--after-stop-hook',help=argparse.SUPPRESS);p.add_argument('--guardian-master',type=int,help=argparse.SUPPRESS);a=p.parse_args()
 if a.guardian_master:return guardian(a.guardian_master)
 if a.dry_run:print(json.dumps(dry_run(),indent=2));return 0
 signal.signal(signal.SIGTERM,stopped_cleanup);signal.signal(signal.SIGINT,stopped_cleanup)
 try:return run(a)
 finally:
  stopped_cleanup()
  if GUARDIAN is not None:
   GUARDIAN.terminate()
   try:GUARDIAN.wait(timeout=2)
   except subprocess.TimeoutExpired:GUARDIAN.kill()
if __name__=='__main__':raise SystemExit(main())
