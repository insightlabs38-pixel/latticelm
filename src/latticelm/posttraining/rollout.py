"""Stochastic rollout records and old-policy parity certification."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Trajectory:
 example_id:str; prompt_ids:tuple[int,...]; generated_ids:tuple[int,...]; sampled_logprobs:tuple[float,...]
 entropies:tuple[float,...]; completion_mask:tuple[bool,...]; termination_reason:str; rng_seed:int
 verifier_result:bool; reward:float; policy_sha256:str
 def to_dict(self):return asdict(self)
def policy_identity(model):
 h=hashlib.sha256()
 for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def sample(model,prompt_ids,example_id,verify,max_new_tokens=16,temperature=1.0,top_k=0,eos_id=None,seed=0,policy_sha256=None):
 if temperature<=0:raise ValueError("temperature must be positive")
 gen=torch.Generator(device="cpu");gen.manual_seed(seed);tokens=list(prompt_ids);generated=[];lps=[];ent=[];reason="max_tokens"
 model.eval()
 with torch.inference_mode():
  for _ in range(max_new_tokens):
   logits=model(torch.tensor(tokens,dtype=torch.long)[None,:])[0][0,-1].float()/temperature
   if top_k:
    values,_=torch.topk(logits,min(top_k,len(logits)));logits=logits.masked_fill(logits<values[-1],float("-inf"))
   logp=F.log_softmax(logits,-1);probs=logp.exp();token=int(torch.multinomial(probs,1,generator=gen));tokens.append(token);generated.append(token);lps.append(float(logp[token]));ent.append(float(-(probs*logp).nan_to_num().sum()))
   if eos_id is not None and token==eos_id:reason="eos";break
 ok=bool(verify(tuple(generated)));return Trajectory(example_id,tuple(prompt_ids),tuple(generated),tuple(lps),tuple(ent),tuple(True for _ in generated),reason,seed,ok,float(ok),policy_sha256 or policy_identity(model))
def recompute_logprobs(model,trajectory,tolerance=1e-5,current_policy_sha256=None):
 ids=list(trajectory.prompt_ids)+list(trajectory.generated_ids);x=torch.tensor(ids[:-1])[None,:]
 with torch.inference_mode():logits=model(x)[0];lp=F.log_softmax(logits.float(),-1)[0]
 start=len(trajectory.prompt_ids)-1;values=tuple(float(lp[start+i,t]) for i,t in enumerate(trajectory.generated_ids));error=max((abs(a-b) for a,b in zip(values,trajectory.sampled_logprobs)),default=0.0)
 return {"values":values,"max_abs_error":error,"pass":error<=tolerance,"policy_match":current_policy_sha256==trajectory.policy_sha256 if current_policy_sha256 else policy_identity(model)==trajectory.policy_sha256}
def certify_trajectory(model,trajectory,tolerance=1e-5,current_policy_sha256=None):
 parity=recompute_logprobs(model,trajectory,tolerance,current_policy_sha256);return {**parity,"masks_valid":len(trajectory.completion_mask)==len(trajectory.generated_ids) and all(trajectory.completion_mask),"reward_attribution":bool(trajectory.example_id) and trajectory.reward==float(trajectory.verifier_result),"status":"PASS" if parity["pass"] and parity["policy_match"] and len(trajectory.completion_mask)==len(trajectory.generated_ids) and all(trajectory.completion_mask) and trajectory.reward==float(trajectory.verifier_result) else "FAIL"}
