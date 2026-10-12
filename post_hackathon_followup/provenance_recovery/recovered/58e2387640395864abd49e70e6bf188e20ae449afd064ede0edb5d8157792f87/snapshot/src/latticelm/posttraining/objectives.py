from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

def completion_mask(prompt_lengths,lengths,width,device=None):
 positions=torch.arange(width,device=device)[None,:];p=torch.as_tensor(prompt_lengths,device=device)[:,None];n=torch.as_tensor(lengths,device=device)[:,None]
 return (positions>=p)&(positions<n)
def masked_causal_loss(logits,targets,mask):
 losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction="none").view_as(targets);den=mask.sum().clamp_min(1);return (losses*mask).sum()/den
def continuation_logprobs(logits,targets,mask):
 token=F.log_softmax(logits,-1).gather(-1,targets[...,None]).squeeze(-1)*mask
 return token.sum(-1),token.sum(-1)/mask.sum(-1).clamp_min(1)
def ranking_cross_entropy(scores,correct_index):return F.cross_entropy(scores,correct_index)
def ranking_margin_loss(scores,correct_index,margin=.2):
 chosen=scores.gather(1,correct_index[:,None]).squeeze(1);wrong=scores.masked_fill(F.one_hot(correct_index,scores.shape[1]).bool(),float("-inf")).max(1).values
 return F.relu(margin-chosen+wrong).mean()
def ranking_objective(raw,normalized,correct_index,mode="ranking_dual",dual_weight=.5,margin_weight=0.,margin=.2):
 if mode not in ("ranking_raw","ranking_norm","ranking_dual"):raise ValueError(mode)
 if not 0<=dual_weight<=1 or margin_weight<0:raise ValueError("invalid ranking coefficients")
 raw_loss=ranking_cross_entropy(raw,correct_index)
 norm_loss=ranking_cross_entropy(normalized,correct_index)
 loss=raw_loss if mode=="ranking_raw" else norm_loss if mode=="ranking_norm" else (1-dual_weight)*raw_loss+dual_weight*norm_loss
 score=raw if mode=="ranking_raw" else normalized if mode=="ranking_norm" else (1-dual_weight)*raw+dual_weight*normalized
 return loss+margin_weight*ranking_margin_loss(score,correct_index,margin)
def joint_objective(completion,ranking,replay=None,rank_weight=.5,replay_weight=.2):
 if rank_weight<0 or replay_weight<0:raise ValueError("invalid joint coefficients")
 return completion+rank_weight*ranking+(0 if replay is None else replay_weight*replay)
def simpo_loss(chosen,rejected,beta=2.0,gamma=.5):return -F.logsigmoid(beta*(chosen-rejected)-gamma).mean()
def grpo_advantages(rewards,group_ids,eps=1e-8):
 out=torch.empty_like(rewards,dtype=torch.float)
 for gid in torch.unique(group_ids):
  ix=group_ids==gid;r=rewards[ix].float();out[ix]=(r-r.mean())/(r.std(unbiased=False)+eps)
 return out
def rloo_advantages(rewards,group_ids):
 out=torch.empty_like(rewards,dtype=torch.float)
 for gid in torch.unique(group_ids):
  ix=group_ids==gid;r=rewards[ix].float();n=len(r);out[ix]=r-(r.sum()-r)/max(n-1,1)
 return out
def policy_gradient_loss(sampled_logprobs,completion_mask_,advantages):
 per=(sampled_logprobs*completion_mask_).sum(-1)/completion_mask_.sum(-1).clamp_min(1);return -(per*advantages.detach()).mean()

@dataclass
class ReplayMixer:
 replay_fraction:float; seed:int=0; draw:int=0
 def __post_init__(self):
  if not 0<=self.replay_fraction<=1:raise ValueError("invalid replay fraction")
 def next_is_replay(self):
  # Low-discrepancy deterministic selector with exact restart state.
  prior=int(self.draw*self.replay_fraction);self.draw+=1;return int(self.draw*self.replay_fraction)>prior
 def state_dict(self):return {"replay_fraction":self.replay_fraction,"seed":self.seed,"draw":self.draw}
 def load_state_dict(self,state):
  if float(state["replay_fraction"])!=self.replay_fraction or int(state["seed"])!=self.seed:raise ValueError("replay identity mismatch")
  self.draw=int(state["draw"])
