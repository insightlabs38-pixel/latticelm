#!/usr/bin/env python3
"""Held-out v2 capability map and continuation calibration."""
import argparse,json,math,time
from collections import defaultdict
from pathlib import Path
import torch
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer
from latticelm.lattice_reason_v2 import generate_example,verify_candidate
from latticelm.posttraining.rollout import sample,policy_identity
from scripts.train_posttraining_worker import candidate_score
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--examples",type=int,default=512);p.add_argument("--max-k",type=int,default=8);p.add_argument("--threads",type=int,default=16);a=p.parse_args();torch.set_num_threads(a.threads);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);cfg=LatticeConfig(**ck["config"]);model=build_model(cfg);model.load_state_dict(ck["model"]);model.eval();tok=load_tokenizer(a.tokenizer);rows=[];started=time.perf_counter()
 identity=policy_identity(model)
 for i in range(a.examples):
  ex=generate_example(i,split="validation",composition_depth=1+i%4);scores=[];means=[]
  for c in ex.candidates:
   total,mean=candidate_score(model,tok,ex.prompt,c,cfg.context_length);scores.append(float(total));means.append(float(mean))
  wrong=max(scores[j] for j in range(len(scores)) if j!=ex.candidate_position);rollouts=[sample(model,tok.encode(ex.prompt)[-240:],ex.global_id,lambda ids:verify_candidate(ex,tok.decode(list(ids)).strip()),8,eos_id=getattr(tok,"eos_id",2),seed=991*i+j,policy_sha256=identity) for j in range(a.max_k)];success=[x.verifier_result for x in rollouts];rows.append({"id":ex.global_id,"skills":ex.skills,"composition_depth":ex.composition_depth,"reasoning_depth":ex.reasoning_depth,"distractors":ex.distractor_count,"correct_sum_loglikelihood":scores[ex.candidate_position],"correct_mean_loglikelihood":means[ex.candidate_position],"best_wrong_margin":scores[ex.candidate_position]-wrong,"candidate_lengths":[len(tok.encode(c)) for c in ex.candidates],"ranking_correct":scores.index(max(scores))==ex.candidate_position,"pass_at_4":any(success[:4]),"pass_at_8":any(success[:8]),"entropy":sum(sum(x.entropies) for x in rollouts)/max(1,sum(len(x.entropies) for x in rollouts)),"rollout_length":sum(len(x.generated_ids) for x in rollouts)/len(rollouts)})
 groups=defaultdict(list)
 for row in rows:groups[("+".join(row["skills"]),row["composition_depth"])].append(row)
 regions={}
 for key,v in groups.items():
  p4=sum(x["pass_at_4"] for x in v)/len(v);p8=sum(x["pass_at_8"] for x in v)/len(v);rank=sum(x["ranking_correct"] for x in v)/len(v);bucket="MASTERED" if p4>=.9 else ("LEARNING_FRONTIER" if p8>=.2 else ("SFT_BOOTSTRAP" if rank>=.25 else "UNREACHABLE_FOR_RL"));regions[str(key)]={"examples":len(v),"ranking_accuracy":rank,"pass_at_4":p4,"pass_at_8":p8,"bucket":bucket}
 result={"schema":"lattice-reason-v2-capability-map-v1","checkpoint":str(a.checkpoint),"regions":regions,"examples":rows,"rollout_tokens_per_second":sum(sum(len(x) for x in [r["candidate_lengths"]]) for r in rows)/max(time.perf_counter()-started,1e-9)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
