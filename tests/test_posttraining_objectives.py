import torch
from latticelm.posttraining.objectives import *
from latticelm.posttraining.rollout import sample,certify_trajectory
class Toy(torch.nn.Module):
 def __init__(self):super().__init__();self.emb=torch.nn.Embedding(7,8);self.head=torch.nn.Linear(8,7,bias=False)
 def forward(self,x):return self.head(self.emb(x)),None
def test_completion_and_padding_masks_zero_prompt_loss():
 mask=completion_mask([2,1],[4,3],5);assert mask.tolist()==[[False,False,True,True,False],[False,True,True,False,False]];logits=torch.randn(2,5,7,requires_grad=True);targets=torch.zeros(2,5,dtype=torch.long);loss=masked_causal_loss(logits,targets,mask);loss.backward();assert logits.grad[~mask].abs().sum()==0
def test_ranking_and_simpo_losses_prefer_correct():
 good=ranking_cross_entropy(torch.tensor([[4.,1.,0.]]),torch.tensor([0]));bad=ranking_cross_entropy(torch.tensor([[0.,4.,1.]]),torch.tensor([0]));assert good<bad and simpo_loss(torch.tensor([3.]),torch.tensor([1.]))<simpo_loss(torch.tensor([1.]),torch.tensor([3.]))
def test_replay_mixer_exact_resume():
 a=ReplayMixer(.4,7);first=[a.next_is_replay() for _ in range(9)];state=a.state_dict();tail=[a.next_is_replay() for _ in range(20)];b=ReplayMixer(.4,7);b.load_state_dict(state);assert tail==[b.next_is_replay() for _ in range(20)] and sum(first+tail)>0
def test_grpo_and_rloo_grouped_advantages():
 rewards=torch.tensor([0.,1.,1.,3.]);groups=torch.tensor([0,0,1,1]);g=grpo_advantages(rewards,groups);r=rloo_advantages(rewards,groups);assert torch.allclose(torch.stack([g[:2].mean(),g[2:].mean()]),torch.zeros(2),atol=1e-6);assert torch.allclose(r,torch.tensor([-1.,1.,-2.,2.]))
def test_policy_gradient_excludes_prompt_and_padding():
 lp=torch.tensor([[9.,1.,2.,9.]],requires_grad=True);mask=torch.tensor([[False,True,True,False]]);loss=policy_gradient_loss(lp,mask,torch.tensor([1.]));loss.backward();assert lp.grad[0,0]==0 and lp.grad[0,3]==0 and lp.grad[0,1]!=0
def test_stochastic_trajectory_parity_rng_and_attribution():
 torch.manual_seed(4);model=Toy();a=sample(model,[1,2],"a",lambda ids:True,4,seed=9);b=sample(model,[1,2],"a",lambda ids:True,4,seed=9);assert a.generated_ids==b.generated_ids and a.sampled_logprobs==b.sampled_logprobs;cert=certify_trajectory(model,a);assert cert["status"]=="PASS" and cert["max_abs_error"]<1e-5
def test_continuation_sum_mean_and_margin():
 logits=torch.zeros(1,3,5);targets=torch.tensor([[1,2,3]]);mask=torch.tensor([[False,True,True]]);total,mean=continuation_logprobs(logits,targets,mask);assert torch.allclose(total,mean*2);assert ranking_margin_loss(torch.tensor([[2.,1.]]),torch.tensor([0]))==0
