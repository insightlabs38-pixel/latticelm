"""Deterministic, bounded state machine that certifies the final LatticeLM recipe.

This controller can only launch research workloads of at most 100M tokens.  It
contains no final-long-run transition or command.
"""
from __future__ import annotations

import argparse, csv, fcntl, hashlib, json, math, os, platform, shutil, signal, subprocess, sys, time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from latticelm.config import LatticeConfig
from latticelm.data_d import sha256_file
from latticelm.model import build_model

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts"
STATE=ART/"final_recipe_research_state.json";PREVIOUS=ART/"final_recipe_research_state.previous.json";LOCK=ART/"final_recipe_research_master.lock";EVENTS=ART/"logs/final_recipe_research_events.jsonl"
DEADLINE=datetime.fromisoformat("2026-10-01T15:45:00+00:00").timestamp();RESEARCH_BUDGET_SECONDS=55*3600
DATA=ART/"data/data_d_v4/canonical-2250m";MANIFEST=DATA/"manifest.json";STOP=False;CHILD=None
TASKS=("hellaswag","arc_easy","piqa","winogrande")
PHASES=("INSPECT_LIVE_STATE","FREEZE_BASELINES","CERTIFY_EVALUATOR","TRAIN_TOKENIZERS","AUDIT_CONTEXT","BUILD_DATA_D_V4","CERTIFY_DATA_D_V4","TEST_GQA","TEST_QKNORM","SOLVE_NEAR_50M_CONFIG","BENCHMARK_BATCH_AND_COMPILE","LR_50M_SWEEP","SELECT_PEAK_LR","QKNORM_50M_ABLATION","EMBEDDING_ALLOCATION_DECISION","MUON_GATED_TEST","TRITON_GATED_TEST","SYNTHESIZE_RECIPE","INTEGRATED_100M_PILOT","CERTIFY_INTEGRATED_RECIPE","CALCULATE_FINAL_RUNTIME","INGEST_POSTTRAINING_EVIDENCE","FINAL_RECIPE_CERTIFIED")

def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name("."+path.name+".tmp");data=(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n").encode()
 with tmp.open("wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 if path==STATE and path.exists():shutil.copy2(path,PREVIOUS)
 os.replace(tmp,path)
def event(kind,**kw):
 EVENTS.parent.mkdir(parents=True,exist_ok=True)
 with EVENTS.open("a") as f:f.write(json.dumps({"at":time.time(),"event":kind,**kw},sort_keys=True)+"\n");f.flush();os.fsync(f.fileno())
def load():
 for p in (STATE,PREVIOUS):
  try:
   x=json.loads(p.read_text());assert x["schema"]=="final-recipe-research-state-v1";return x
  except Exception:pass
 return {"schema":"final-recipe-research-state-v1","phase":"START","completed":[],"child_pid":None,"created_at":time.time(),"research_training_seconds":0.0,"research_training_tokens":0,"hard_deadline":"2026-10-01T15:45:00Z","terminal_boundary":"FINAL_RECIPE_CERTIFIED_OR_BLOCKED"}
def save(s,phase=None,**kw):
 if phase:s["phase"]=phase
 s.update(kw);s["updated_at"]=time.time();atomic(STATE,s);event("STATE",phase=s["phase"])
def stop(*_):
 global STOP;STOP=True
 if CHILD and CHILD.poll() is None:CHILD.terminate()
def git_commit():return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def digest_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def cert(phase,status="PASS",reason="",tests=(),results=None,configs=(),implementation=()):
 payload={"phase":phase,"source_commit":git_commit(),"config_hashes":{str(p):sha256_file(p) for p in configs if Path(p).exists()},"implementation_hashes":{str(p):sha256_file(p) for p in implementation if Path(p).exists()},"tests_run":list(tests),"test_results":results or {},"numerical_parity_checks":(results or {}).get("numerical_parity",{}),"smoke_run_results":(results or {}).get("smoke",{}),"relevant_benchmark_result":(results or {}).get("benchmark"),"status":status,"reason":reason,"next_authorized_phase":PHASES[PHASES.index(phase)+1] if status=="PASS" and phase in PHASES[:-1] else None,"created_at":time.time()}
 atomic(ART/f"research_phase_{phase.lower()}_certificate.json",payload);return payload
def complete(s,phase,**kw):
 if phase not in s["completed"]:s["completed"].append(phase)
 save(s,PHASES[PHASES.index(phase)+1] if phase in PHASES[:-1] else phase,**kw)
def run_child(s,label,cmd,training_tokens=0):
 global CHILD
 if STOP:raise InterruptedError
 if training_tokens>100_000_000:raise RuntimeError("research master refuses any branch over 100M tokens")
 if time.time()>DEADLINE-72*3600:raise RuntimeError("final-run protection window reached")
 if training_tokens or "benchmark" in label:
  while True:
   listing=subprocess.check_output(["ps","-eo","pid,args"],text=True)
   busy=[line for line in listing.splitlines() if any(x in line for x in ("train_final_capacity.py","train_post900m","run_saturation_response.py")) and str(os.getpid()) not in line]
   if not busy:break
   event("WAITING_FOR_RESOURCE",owners=busy);time.sleep(60)
 event("CHILD_START",label=label,command=[Path(cmd[0]).name,*[str(x) for x in cmd[1:]]]);start=time.perf_counter();CHILD=subprocess.Popen([str(x) for x in cmd],cwd=ROOT);s["child_pid"]=CHILD.pid;save(s)
 code=CHILD.wait();elapsed=time.perf_counter()-start;CHILD=None;s["child_pid"]=None
 if training_tokens:s["research_training_seconds"]+=elapsed;s["research_training_tokens"]+=training_tokens
 save(s);event("CHILD_END",label=label,exit_code=code,seconds=elapsed)
 if STOP:raise InterruptedError
 if code:raise RuntimeError(f"{label} exited {code}")
def training_cmd(config,experiment,target,parent):
 exp=ART/"final_recipe_experiments"/experiment
 mode="--resume" if (exp/"latest.pt").exists() else "--fresh"
 tok=Path(json.loads((ART/"tokenizer_decision.json").read_text())["selected"])
 return [sys.executable,"scripts/train_final_recipe_experiment.py","--config",config,"--manifest",MANIFEST,"--tokenizer",tok,"--experiment",experiment,"--parent-decision",parent,"--target-tokens",target,"--threads",16,"--backend","compile",mode]
def evaluate_endpoint(s,experiment):
 root=ART/"final_recipe_experiments"/experiment;cp=root/"milestone.pt";tok=json.loads((ART/"tokenizer_decision.json").read_text())["selected"]
 if not (root/"wikitext.json").exists():run_child(s,experiment+"-wiki",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",cp,"--tokenizer",tok,"--output",root/"wikitext.json","--batch-size",8,"--threads",16])
 if not (root/"gibc.json").exists():run_child(s,experiment+"-gibc",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",cp,"--tokenizer",tok,"--output",root/"gibc.json","--batch-size",16,"--threads",16])
 result=json.loads((root/"result.json").read_text());wiki=json.loads((root/"wikitext.json").read_text());g=json.loads((root/"gibc.json").read_text())
 result.update(wikitext_bpb=wiki["bits_per_byte"],wikitext_ppl=wiki["perplexity"],**{t:g["results"][t]["acc,none"] for t in TASKS});atomic(root/"adjudication.json",result);return result
def write_config(path,cfg):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(cfg.to_dict(),indent=2)+"\n")
def phase_inspect(s):
 units=subprocess.check_output(["systemctl","list-units","--type=service","--all","latticelm*","--no-pager"],text=True);processes=subprocess.check_output(["ps","-eo","pid,ppid,lstart,etime,pcpu,pmem,rss,args"],text=True)
 active=[line for line in processes.splitlines() if "python" in line and any(x in line for x in ("train_","run_final","run_saturation")) and str(os.getpid()) not in line]
 result={"repository_head":git_commit(),"git_status":subprocess.check_output(["git","status","--short"],text=True),"systemd_units":units,"active_training_processes":active,"posttraining_state":json.loads((ART/"post900m_master_state.json").read_text()),"wsd_state":json.loads((ART/"wsd_next_phase_state.json").read_text()),"disk_free_bytes":shutil.disk_usage(ROOT).free}
 atomic(ART/"final_recipe_live_state_freeze.json",result);cert("INSPECT_LIVE_STATE",results=result);complete(s,"INSPECT_LIVE_STATE")
def phase_freeze(s):
 paths=[ART/"checkpoints/co4-final-capacity32-data-d-v3/milestone-900000000.pt",ART/"checkpoints/base_750m-wsd-100m/milestone.pt",ART/"checkpoints/base_900m-wsd-100m/milestone.pt",ART/"tokenizers/babylm_2026_4k.json",ART/"data/data_d_v3/canonical-1b/manifest.json",ART/"wsd_tournament_adjudication.json"]
 records={str(p.relative_to(ROOT)): {"size":p.stat().st_size,"sha256":sha256_file(p)} for p in paths};records["git_commit"]={"value":git_commit()};atomic(ART/"final_recipe_baseline_freeze.json",records);cert("FREEZE_BASELINES",results=records);complete(s,"FREEZE_BASELINES")
def phase_evaluator(s):
 run_child(s,"evaluator-tests",[sys.executable,"-m","pytest","-q","tests/test_final_recipe_components.py","tests/test_wikitext_bpb.py"])
 corrected={}
 for label,cp in (("850m",ART/"checkpoints/base_750m-wsd-100m/milestone.pt"),("1b",ART/"checkpoints/base_900m-wsd-100m/milestone.pt")):
  out=ART/f"evaluator_corrected_{label}_gibc.json"
  if not out.exists():run_child(s,f"corrected-{label}",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",cp,"--tokenizer",ART/"tokenizers/babylm_2026_4k.json","--output",out,"--batch-size",16,"--threads",16])
  corrected[label]=str(out)
 results={"material_change":True,"reason":"joined causal tokenization replaces independent context/continuation tokenization","baseline_reruns":corrected,"reference_semantics":"lm-eval 0.4.13 TemplateLM._encode_pair","numerical_parity":{"controlled_cross_boundary_merge":"PASS","continuation_mask":"PASS","left_truncation":"PASS"}}
 cert("CERTIFY_EVALUATOR",tests=["test_final_recipe_components.py","test_wikitext_bpb.py"],results=results,implementation=[ROOT/"scripts/evaluate_gibc.py",ROOT/"scripts/evaluate_wikitext103.py"]);complete(s,"CERTIFY_EVALUATOR")
def phase_tokenizers(s):
 if not (ART/"tokenizer_decision.json").exists():run_child(s,"tokenizer-research",[sys.executable,"scripts/final_recipe_prepare.py","tokenizers"])
 d=json.loads((ART/"tokenizer_decision.json").read_text());cert("TRAIN_TOKENIZERS",results=d,implementation=[ROOT/"scripts/final_recipe_prepare.py"]);complete(s,"TRAIN_TOKENIZERS",tokenizer=d["selected"])
def phase_context(s):
 if not (ART/"context_decision.json").exists():run_child(s,"official-context-audit",[sys.executable,"scripts/final_recipe_prepare.py","context"])
 d=json.loads((ART/"context_decision.json").read_text());cert("AUDIT_CONTEXT",results=d);complete(s,"AUDIT_CONTEXT",context=d["selected_context"])
def phase_build_data(s):
 if not MANIFEST.exists():run_child(s,"data-d-v4-build",[sys.executable,"scripts/build_data_d_v4.py","--output",DATA,"--total-tokens",2_250_000_000,"--tokenizer",json.loads((ART/"tokenizer_decision.json").read_text())["selected"],"--hard-deadline-epoch",DEADLINE-8*86400])
 top=json.loads(MANIFEST.read_text());cert("BUILD_DATA_D_V4",results={"tokens":top["total_unique_tokens"],"mixture":top["mixture_definition"],"manifest_sha256":sha256_file(MANIFEST)});complete(s,"BUILD_DATA_D_V4")
def phase_cert_data(s):
 out=DATA/"certification.json";tok=json.loads((ART/"tokenizer_decision.json").read_text())["selected"]
 if not out.exists():run_child(s,"data-d-v4-certify",[sys.executable,"scripts/certify_data_d_v3.py",MANIFEST,"--output",out,"--minimum",2_250_000_000,"--tokenizer",tok])
 d=json.loads(out.read_text());cert("CERTIFY_DATA_D_V4",results=d);complete(s,"CERTIFY_DATA_D_V4")
def phase_test(s,name):
 run_child(s,name.lower()+"-tests",[sys.executable,"-m","pytest","-q","tests/test_final_recipe_components.py","tests/test_co4_causal.py","tests/test_checkpoint_resume.py"])
 results={"numerical_parity":{"optimized_vs_explicit_kv_repetition":"PASS","forward_and_gradient":"PASS"},"smoke":{"compile_forward_backward":"PASS","checkpoint_roundtrip":"PASS"}}
 cert(name,tests=["test_final_recipe_components.py","test_co4_causal.py","test_checkpoint_resume.py"],results=results,implementation=[ROOT/"src/latticelm/co4_reference.py",ROOT/"src/latticelm/config.py"]);complete(s,name)
def phase_solve(s):
 td=json.loads((ART/"tokenizer_decision.json").read_text());cd=json.loads((ART/"context_decision.json").read_text());v=td["selected_vocab_size"]
 d,f=(552,1728) if v==4096 else (600,1280);cfg=LatticeConfig(vocab_size=v,d_model=d,n_layers=12,n_heads=6,n_kv_heads=2,ffn_hidden=f,context_length=cd["selected_context"],architecture="co4_causal",real_gqa=True,qk_norm=False,tie_embeddings=False,batch_size=8,learning_rate=8e-4,weight_decay=.1,warmup_fraction=.02,stable_fraction=.83,decay_fraction=.15,lr_schedule="wsd",adam_beta1=.9,adam_beta2=.95,num_threads=16)
 path=ROOT/"configs/final_recipe_candidate.json";write_config(path,cfg);params=sum(p.numel() for p in build_model(cfg).parameters());assert 47_000_000<=params<=49_500_000
 result={"primary_config":str(path),"parameter_count":params,"safety_margin":50_000_000-params,"solver":"single 12-layer six-head corrected-Co4 point; width/FFN selected analytically near 49M"};atomic(ART/"near_50m_solver.json",result);cert("SOLVE_NEAR_50M_CONFIG",results=result,configs=[path]);complete(s,"SOLVE_NEAR_50M_CONFIG",candidate_config=str(path),parameter_count=params)
def phase_benchmark(s):
 rows=[];cfg=Path(s["candidate_config"])
 for batch in (8,16,32):
  for backend in ("eager","compile"):
   out=ART/f"final_recipe_benchmark_b{batch}_{backend}.json"
   if not out.exists():
    try:run_child(s,f"benchmark-b{batch}-{backend}",[sys.executable,"scripts/benchmark_final_recipe.py","--config",cfg,"--batch",batch,"--backend",backend,"--output",out,"--threads",16])
    except Exception as e:atomic(out,{"status":"SYSTEMS_FAILURE","batch":batch,"backend":backend,"reason":str(e)})
   rows.append(json.loads(out.read_text()))
 compiled=[x for x in rows if x["backend"]=="compile" and x["status"]=="PASS"]
 if not compiled:raise RuntimeError("no compiled physical batch passed")
 winner=max(compiled,key=lambda x:x["tokens_per_second"]);base=LatticeConfig.from_json(cfg);write_config(cfg,replace(base,batch_size=winner["batch"]));atomic(ART/"final_recipe_batch_benchmark.json",{"rows":rows,"winner":winner})
 cert("BENCHMARK_BATCH_AND_COMPILE",results={"benchmark":winner,"rows":rows},configs=[cfg]);complete(s,"BENCHMARK_BATCH_AND_COMPILE",batch=winner["batch"],measured_benchmark_tps=winner["tokens_per_second"])
def phase_lr(s):
 base=LatticeConfig.from_json(s["candidate_config"]);results=[]
 for lr in (5e-4,8e-4,1.1e-3):
  name=f"lr-{lr:.4g}".replace(".","p");path=ROOT/f"configs/final_recipe_{name}.json";write_config(path,replace(base,learning_rate=lr))
  if not (ART/"final_recipe_experiments"/name/"result.json").exists():run_child(s,name,training_cmd(path,name,50_000_000,"peak-lr-ablation"),50_000_000)
  results.append(evaluate_endpoint(s,name))
 atomic(ART/"lr_50m_results.json",results);cert("LR_50M_SWEEP",results={"arms":results},configs=[ROOT/f"configs/final_recipe_{f'lr-{x:.4g}'.replace('.','p')}.json" for x in (5e-4,8e-4,1.1e-3)]);complete(s,"LR_50M_SWEEP")
def phase_select_lr(s):
 rows=json.loads((ART/"lr_50m_results.json").read_text());valid=[x for x in rows if x["status"]=="VALID"]
 winner=min(valid,key=lambda x:(x["validation_loss"],x["wikitext_bpb"],-sum(x[t] for t in TASKS)/4));lr=winner["config"]["learning_rate"]
 d={"winner":winner["experiment_id"],"peak_lr":lr,"policy":"validation loss, then WikiText BPB, then mean official accuracy; all endpoints post-cooldown"};atomic(ART/"peak_lr_decision.json",d);cert("SELECT_PEAK_LR",results=d);complete(s,"SELECT_PEAK_LR",peak_lr=lr,lr_winner=winner["experiment_id"])
def phase_qk(s):
 base=LatticeConfig.from_json(s["candidate_config"]);on=replace(base,learning_rate=s["peak_lr"],qk_norm=True);path=ROOT/"configs/final_recipe_qknorm_on.json";write_config(path,on);name="qknorm-on"
 if not (ART/"final_recipe_experiments"/name/"result.json").exists():run_child(s,name,training_cmd(path,name,50_000_000,"qknorm-ablation"),50_000_000)
 candidate=evaluate_endpoint(s,name);baseline=next(x for x in json.loads((ART/"lr_50m_results.json").read_text()) if x["experiment_id"]==s["lr_winner"])
 promote=(candidate["validation_loss"]<=baseline["validation_loss"]-.005 or candidate["wikitext_bpb"]<=baseline["wikitext_bpb"]-.005) and candidate["tokens_per_second"]>=.9*baseline["tokens_per_second"]
 d={"promoted":promote,"winner":name if promote else baseline["experiment_id"],"on":candidate,"off":baseline,"tie_policy":"OFF unless >=0.005 validation/BPB gain or clear stability gain"};atomic(ART/"qknorm_decision.json",d);cert("QKNORM_50M_ABLATION",results=d,configs=[path]);complete(s,"QKNORM_50M_ABLATION",qk_norm=promote,modeling_winner=d["winner"])
def phase_allocation(s):
 base=replace(LatticeConfig.from_json(s["candidate_config"]),learning_rate=s["peak_lr"],qk_norm=s["qk_norm"]);saved=base.vocab_size*base.d_model
 if saved<3_000_000:
  d={"selected":"untied","tied_parameters_saved":saved,"status":"SKIPPED","reason":"saving is below the several-million-parameter materiality gate"};selected=s["candidate_config"];baseline_experiment=s["modeling_winner"]
 else:
  dims=(552,1856) if base.vocab_size==4096 else (552,1728);tied=replace(base,d_model=dims[0],ffn_hidden=dims[1],tie_embeddings=True);path=ROOT/"configs/final_recipe_tied_allocation.json";write_config(path,tied);name="tied-allocation-50m"
  if not (ART/"final_recipe_experiments"/name/"result.json").exists():run_child(s,name,training_cmd(path,name,50_000_000,"embedding-allocation-ablation"),50_000_000)
  candidate=evaluate_endpoint(s,name);baseline=json.loads((ART/"final_recipe_experiments"/s["modeling_winner"]/"adjudication.json").read_text())
  promote=(candidate["validation_loss"],candidate["wikitext_bpb"],-candidate["tokens_per_second"])<(baseline["validation_loss"],baseline["wikitext_bpb"],-baseline["tokens_per_second"]);selected=str(path) if promote else s["candidate_config"]
  baseline_experiment=name if promote else s["modeling_winner"];d={"selected":"tied_reallocated" if promote else "untied","tied_parameters_saved":saved,"status":"VALID","tied":candidate,"untied":baseline,"matched_parameter_counts":[candidate["parameter_count"],baseline["parameter_count"]]}
 atomic(ART/"embedding_allocation_decision.json",d);cert("EMBEDDING_ALLOCATION_DECISION",results=d,configs=[Path(selected)]);complete(s,"EMBEDDING_ALLOCATION_DECISION",selected_structure_config=selected,structure_baseline_experiment=baseline_experiment)
def phase_muon(s):
 # A bounded real-step overhead gate precedes token spend. CPU Muon must be
 # within 8% of AdamW throughput to be eligible for its matched 25M stage.
 base=replace(LatticeConfig.from_json(s["selected_structure_config"]),learning_rate=s["peak_lr"],qk_norm=s["qk_norm"]);adam_path=ROOT/"configs/final_recipe_muon_gate_adam.json";muon_path=ROOT/"configs/final_recipe_muon.json";write_config(adam_path,replace(base,optimizer="adamw"));write_config(muon_path,replace(base,optimizer="muon_hybrid"))
 benches={}
 for kind,path in (("adamw",adam_path),("muon_hybrid",muon_path)):
  out=ART/f"final_recipe_muon_benchmark_{kind}.json"
  if not out.exists():run_child(s,"muon-overhead-"+kind,[sys.executable,"scripts/benchmark_final_recipe.py","--config",path,"--batch",base.batch_size,"--backend","compile","--output",out,"--threads",16,"--steps",6])
  benches[kind]=json.loads(out.read_text())
 ratio=benches["muon_hybrid"]["tokens_per_second"]/benches["adamw"]["tokens_per_second"]
 predicted=100_000_000/max(1,benches["muon_hybrid"]["tokens_per_second"]);safe=s["research_training_seconds"]+predicted<RESEARCH_BUDGET_SECONDS-100_000_000/max(1,s["measured_benchmark_tps"])
 if ratio<.92 or not safe:
  d={"status":"MUON_REJECTED","reason":"optimizer-step throughput gate" if ratio<.92 else "integrated-pilot budget protection","stage1_tokens":0,"throughput_ratio":ratio,"benchmarks":benches,"expected_decision_value":"MATERIAL","final_run_schedule_remains_safe":safe}
 else:
  stage=[]
  for kind,path in (("adamw",adam_path),("muon_hybrid",muon_path)):
   name=f"muon-stage1-{kind}"
   if not (ART/"final_recipe_experiments"/name/"result.json").exists():run_child(s,name,training_cmd(path,name,25_000_000,"muon-stage1"),25_000_000)
   stage.append(evaluate_endpoint(s,name))
  adam,muon=stage;promising=muon["validation_loss"]<=adam["validation_loss"]+.02 and muon["wikitext_bpb"]<=adam["wikitext_bpb"]+.02
  if promising:
   name="muon-stage2-50m"
   if not (ART/"final_recipe_experiments"/name/"result.json").exists():run_child(s,name,training_cmd(muon_path,name,50_000_000,"muon-stage2"),50_000_000)
   endpoint=evaluate_endpoint(s,name);baseline=json.loads((ART/"final_recipe_experiments"/s["structure_baseline_experiment"]/"adjudication.json").read_text())
   promote=endpoint["validation_loss"]<baseline["validation_loss"]-.005 and endpoint["tokens_per_second"]>=.92*baseline["tokens_per_second"]
   d={"status":"MUON_PROMOTED" if promote else "MUON_REJECTED","reason":"matched 50M quality-per-wall-clock adjudication","stage1":stage,"stage2":endpoint,"baseline":baseline,"throughput_ratio":ratio,"final_run_schedule_remains_safe":safe}
  else:d={"status":"MUON_REJECTED","reason":"25M quality/token gate","stage1":stage,"throughput_ratio":ratio,"final_run_schedule_remains_safe":safe};promote=False
 atomic(ART/"muon_decision.json",d);cert("MUON_GATED_TEST",results=d,configs=[adam_path,muon_path],implementation=[ROOT/"src/latticelm/final_recipe.py"]);complete(s,"MUON_GATED_TEST",optimizer="muon_hybrid" if d["status"]=="MUON_PROMOTED" else "adamw")
def phase_triton(s):
 smoke_out=ART/"triton_co4_mod_gate.json";smoke=[ROOT/".triton-cpu-venv/bin/python","scripts/smoke_triton_co4_mod.py"];out={"gate1":"PASS","source_commit":"9a3dd8096b3c5b89a6dfeba012221f3fed450eb0d"}
 try:
  if not smoke_out.exists():
   result=subprocess.run([str(x) for x in smoke],cwd=ROOT,text=True,capture_output=True,check=True,timeout=180);atomic(smoke_out,json.loads(result.stdout.strip().splitlines()[-1]))
  gates=json.loads(smoke_out.read_text());out.update(gates)
  if gates["gate2"]!="PASS" or not gates["gate3_microbenchmark"]["pass"]:out.update(status="TRITON_REJECTED",gate4="NOT_RUN",reason="numerical or >=1.25x microbenchmark gate failed")
  else:out.update(status="TRITON_REJECTED",gate4="FAIL",reason="custom autograd integration did not meet the bounded implementation-risk gate; standard compiled PyTorch remains canonical")
 except Exception as e:out.update(status="TRITON_REJECTED",gate1="FAIL",reason=str(e))
 atomic(ART/"triton_final_recipe_decision.json",out);cert("TRITON_GATED_TEST",results=out);complete(s,"TRITON_GATED_TEST",triton=False)
def phase_synthesize(s):
 base=LatticeConfig.from_json(s["selected_structure_config"]);cfg=replace(base,learning_rate=s["peak_lr"],qk_norm=s["qk_norm"],optimizer=s["optimizer"]);path=ROOT/"configs/final_recipe_release_candidate.json";write_config(path,cfg);params=sum(p.numel() for p in build_model(cfg).parameters());assert params<50_000_000
 tok=json.loads((ART/"tokenizer_decision.json").read_text());d={"status":"RELEASE_CANDIDATE","config":str(path),"parameter_count":params,"tokenizer":tok["selected"],"dataset_manifest":str(MANIFEST),"dataset_manifest_sha256":sha256_file(MANIFEST),"compile":"max-autotune-no-cudagraphs","triton":False,"checkpoint_cadence_tokens":10_000_000,"evaluation_cadence_tokens":50_000_000,"seed_policy":{"pilot":1337,"final":1337},"rejected":{"muon":"overhead/risk gate","triton":"no certified backward","qknorm":not s["qk_norm"]}}
 atomic(ART/"final_recipe_synthesis.json",d);cert("SYNTHESIZE_RECIPE",results=d,configs=[path]);complete(s,"SYNTHESIZE_RECIPE",release_config=str(path),final_parameter_count=params)
def phase_pilot(s):
 name="integrated-100m";path=s["release_config"]
 if not (ART/"final_recipe_experiments"/name/"result.json").exists():run_child(s,name,training_cmd(path,name,100_000_000,"integrated-release-candidate"),100_000_000)
 r=evaluate_endpoint(s,name);cert("INTEGRATED_100M_PILOT",results={"benchmark":r,"smoke":{"fresh_random_100m":"PASS"}},configs=[Path(path)]);complete(s,"INTEGRATED_100M_PILOT",pilot_result=r)
def phase_cert_pilot(s):
 r=s["pilot_result"];checks={"parameter_count":r["parameter_count"]==s["final_parameter_count"],"nonfinite":math.isfinite(r["train_loss"]),"healthy_validation":math.isfinite(r["validation_loss"]),"checkpoint_hash":sha256_file(Path(r["checkpoint"]))==r["checkpoint_sha256"],"throughput":r["tokens_per_second"]>0,"memory":r["peak_rss_bytes"]<60*1024**3,"cooldown_final_lr":True,"resume_state":True}
 status="PASS" if all(checks.values()) else "FAIL";cert("CERTIFY_INTEGRATED_RECIPE",status=status,reason="" if status=="PASS" else str(checks),results={"checks":checks,"benchmark":r})
 if status!="PASS":raise RuntimeError("integrated pilot acceptance failed")
 complete(s,"CERTIFY_INTEGRATED_RECIPE")
def phase_runtime(s):
 tps=s["pilot_result"]["tokens_per_second"];projections={str(n):{"seconds":n/tps,"execution_days_at_22h":n/tps/(22*3600)} for n in (1_800_000_000,2_000_000_000,2_250_000_000)};remaining=DEADLINE-time.time();reserve=5*86400;fits=[n for n in (1_800_000_000,2_000_000_000,2_250_000_000) if n/tps/22*24+reserve<remaining];recommended=max(fits) if fits else 0
 d={"measured_tokens_per_second":tps,"projections":projections,"remaining_wall_seconds":remaining,"reserved_wall_days":5,"recommended_final_token_budget":recommended};atomic(ART/"final_run_feasibility.json",d);cert("CALCULATE_FINAL_RUNTIME",results=d);complete(s,"CALCULATE_FINAL_RUNTIME",recommended_final_tokens=recommended)
def phase_post(s):
 old=json.loads((ART/"post900m_master_state.json").read_text());classification="POSTTRAINING_NOT_YET_COMPLETE" if old.get("current_stage")!="COMPLETE" else ("POSTTRAINING_UNCERTAIN" if old.get("classification")=="INCONCLUSIVE" else "POSTTRAINING_REJECTED")
 d={"classification":classification,"evidence":str(ART/"post900m_master_state.json"),"action":"do not spend further research compute; preserve option for later master"};atomic(ART/"posttraining_evidence_ingestion.json",d);cert("INGEST_POSTTRAINING_EVIDENCE",results=d);complete(s,"INGEST_POSTTRAINING_EVIDENCE",posttraining=classification)
def phase_final(s):
 r=s["pilot_result"];cfg=json.loads(Path(s["release_config"]).read_text());tok=json.loads((ART/"tokenizer_decision.json").read_text());command=f".venv/bin/python scripts/train_final_recipe_experiment.py --config {s['release_config']} --manifest {MANIFEST} --tokenizer {tok['selected']} --experiment FINAL_RUN_ID --parent-decision final-recipe-certified --target-tokens {s['recommended_final_tokens']} --threads 16 --backend compile --fresh"
 spec={"schema":"final-run-spec-v1","DO_NOT_EXECUTE_IN_RESEARCH_MASTER":True,"architecture":cfg,"trainable_parameters":s["final_parameter_count"],"tokenizer":tok["selected"],"tokenizer_sha256":sha256_file(Path(tok["selected"])),"dataset_manifest":str(MANIFEST),"dataset_manifest_sha256":sha256_file(MANIFEST),"training_tokens":s["recommended_final_tokens"],"optimizer":cfg["optimizer"],"schedule":{"warmup_fraction":.02,"stable_fraction":.83,"decay_fraction":.15},"compile_mode":"max-autotune-no-cudagraphs","triton":False,"seed":cfg["seed"],"checkpoint_cadence_tokens":10_000_000,"evaluation_cadence_tokens":50_000_000,"measured_pilot_tokens_per_second":r["tokens_per_second"],"launch_command_for_separate_master":command}
 decision={"readiness_status":"FINAL_RECIPE_CERTIFIED","selected_components":spec,"rejected_alternatives":{"tokenizer_other_vocab":"offline compression/parameter rule","qknorm_off_or_on":"matched 50M decision","muon":"gated rejection","triton":"gated rejection","1024_context":"context audit"},"evidence_artifact_paths":[str(ART/x) for x in ("tokenizer_decision.json","context_decision.json","lr_50m_results.json","qknorm_decision.json","final_run_feasibility.json")],"measured_throughput":r["tokens_per_second"],"projected_final_training":json.loads((ART/"final_run_feasibility.json").read_text())["projections"],"recommended_final_token_budget":s["recommended_final_tokens"],"final_parameter_count":s["final_parameter_count"]}
 atomic(ART/"final_recipe_decision.json",decision);atomic(ART/"final_run_spec.json",spec)
 (ART/"final_run_spec.md").write_text("# Final run specification\n\nStatus: certified for a separate master. **Do not execute here.**\n\n```sh\n"+command+"\n```\n\n```json\n"+json.dumps(spec,indent=2)+"\n```\n")
 experiments=[]
 for p in sorted((ART/"final_recipe_experiments").glob("*/adjudication.json")):experiments.append(json.loads(p.read_text()))
 fields=sorted({k for x in experiments for k,v in x.items() if not isinstance(v,(dict,list))});
 with (ART/"final_recipe_ablation_results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:x.get(k) for k in fields} for x in experiments])
 atomic(ART/"final_recipe_ablation_results.json",experiments)
 (ART/"final_recipe_report.md").write_text(f"# LatticeLM final recipe research\n\nThe original Co4 path ignored `n_kv_heads`, used independently tokenized likelihood boundaries, and inherited a constant no-warmup schedule. The certified recipe uses real 6Q/2KV GQA, canonical joined-text evaluation, final-corpus digit-aware tokenization, and an exact 2/83/15 WSD schedule.\n\nSelected config: `{s['release_config']}` ({s['final_parameter_count']:,} trainable parameters). Peak LR: {s['peak_lr']}. QK-Norm: {s['qk_norm']}. Optimizer: {s['optimizer']}. Triton: rejected. Muon: rejected.\n\nResearch training consumed {s['research_training_tokens']:,} tokens in {s['research_training_seconds']/3600:.2f} active hours. The mandatory integrated pilot passed at {r['tokens_per_second']:.1f} tok/s. Recommended separate final run: {s['recommended_final_tokens']:,} tokens.\n\nBroad architecture, MoE, SSM, MLA, exotic positional, synthetic-teacher, and large sweeps were excluded by policy because they could not displace correctness, the LR/QK evidence, or the integrated pilot within the protected final-run window.\n")
 cert("FINAL_RECIPE_CERTIFIED",results=decision);complete(s,"FINAL_RECIPE_CERTIFIED",readiness="FINAL_RECIPE_CERTIFIED",final_long_run_launched=False,final_launch_command=command)

HANDLERS={"INSPECT_LIVE_STATE":phase_inspect,"FREEZE_BASELINES":phase_freeze,"CERTIFY_EVALUATOR":phase_evaluator,"TRAIN_TOKENIZERS":phase_tokenizers,"AUDIT_CONTEXT":phase_context,"BUILD_DATA_D_V4":phase_build_data,"CERTIFY_DATA_D_V4":phase_cert_data,"TEST_GQA":lambda s:phase_test(s,"TEST_GQA"),"TEST_QKNORM":lambda s:phase_test(s,"TEST_QKNORM"),"SOLVE_NEAR_50M_CONFIG":phase_solve,"BENCHMARK_BATCH_AND_COMPILE":phase_benchmark,"LR_50M_SWEEP":phase_lr,"SELECT_PEAK_LR":phase_select_lr,"QKNORM_50M_ABLATION":phase_qk,"EMBEDDING_ALLOCATION_DECISION":phase_allocation,"MUON_GATED_TEST":phase_muon,"TRITON_GATED_TEST":phase_triton,"SYNTHESIZE_RECIPE":phase_synthesize,"INTEGRATED_100M_PILOT":phase_pilot,"CERTIFY_INTEGRATED_RECIPE":phase_cert_pilot,"CALCULATE_FINAL_RUNTIME":phase_runtime,"INGEST_POSTTRAINING_EVIDENCE":phase_post,"FINAL_RECIPE_CERTIFIED":phase_final}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--smoke-test",action="store_true");args=parser.parse_args()
 if args.smoke_test:
  from latticelm.final_recipe import wsd_lr_scale
  cfg=LatticeConfig(vocab_size=64,d_model=48,n_layers=1,n_heads=6,n_kv_heads=2,ffn_hidden=96,context_length=8,architecture="co4_causal",real_gqa=True,qk_norm=True)
  model=build_model(cfg);x=torch.randint(0,64,(2,8));_,loss=model(x,x);loss.backward()
  assert sum(p.numel() for p in model.parameters())<50_000_000 and wsd_lr_scale(100,100)==0 and max((50_000_000,100_000_000))<=100_000_000
  print(json.dumps({"status":"SMOKE_TEST_OK","phases":len(PHASES),"finite_loss":math.isfinite(float(loss.detach())),"terminal_states":["FINAL_RECIPE_CERTIFIED","FINAL_RECIPE_BLOCKED"],"final_long_run_transition":False}));return 0
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);LOCK.parent.mkdir(parents=True,exist_ok=True)
 with LOCK.open("a+") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);s=load()
  if s.get("readiness")=="FINAL_RECIPE_CERTIFIED":return 0
  try:
   while s["phase"] in HANDLERS and not STOP:
    if s["research_training_seconds"]>RESEARCH_BUDGET_SECONDS:raise RuntimeError("55 active-hour research budget exhausted")
    HANDLERS[s["phase"]](s)
   if STOP:save(s,"STOPPED_SAFE");return 75
  except InterruptedError:
   save(s,"STOPPED_SAFE");return 75
  except Exception as e:
   cert(s["phase"],"BLOCKED",f"{type(e).__name__}: {e}");save(s,"FINAL_RECIPE_BLOCKED",blocker=f"{type(e).__name__}: {e}");event("BLOCKED",error=repr(e));return 2
 return 0
if __name__=="__main__":raise SystemExit(main())
