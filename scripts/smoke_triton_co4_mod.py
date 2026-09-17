"""Isolated Triton-CPU parity and microbenchmark gate for Co4 MOD."""
from __future__ import annotations
import json,time
import torch
import torch.nn.functional as F
import triton,triton.language as tl

@triton.jit
def forward_kernel(r,c,out,n,BLOCK:tl.constexpr):
 o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);m=o<n;rv=tl.load(r+o,mask=m);cv=tl.load(c+o,mask=m);z=rv*rv+2*rv+cv*(1+tl.abs(rv));tl.store(out+o,tl.minimum(6.,tl.maximum(0.,z)),mask=m)
@triton.jit
def backward_kernel(r,c,g,gr,gc,n,BLOCK:tl.constexpr):
 o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);m=o<n;rv=tl.load(r+o,mask=m);cv=tl.load(c+o,mask=m);gv=tl.load(g+o,mask=m);z=rv*rv+2*rv+cv*(1+tl.abs(rv));active=(z>0)&(z<6);sgn=tl.where(rv>0,1.,tl.where(rv<0,-1.,0.));tl.store(gc+o,gv*active*(1+tl.abs(rv)),mask=m);tl.store(gr+o,gv*active*(2*rv+2+cv*sgn),mask=m)
def launch(r,c,g=None):
 n=r.numel();grid=(triton.cdiv(n,1024),);out=torch.empty_like(r);forward_kernel[grid](r,c,out,n=n,BLOCK=1024)
 if g is None:return out
 gr=torch.empty_like(r);gc=torch.empty_like(c);backward_kernel[grid](r,c,g,gr,gc,n=n,BLOCK=1024);return out,gr,gc
def main():
 triton.runtime.driver.set_active_to_cpu();cases=[]
 for n in (1024,65536):
  torch.manual_seed(n);r=torch.randn(n);c=torch.randn(n);g=torch.randn(n);out,gr,gc=launch(r,c,g);rr=r.clone().requires_grad_();cc=c.clone().requires_grad_();ref=F.relu6(rr.square()+2*rr+cc*(1+rr.abs()));ref.backward(g)
  cases.append({"n":n,"forward_max_abs":float((out-ref).abs().max()),"receptive_grad_max_abs":float((gr-rr.grad).abs().max()),"context_grad_max_abs":float((gc-cc.grad).abs().max())})
 edge=torch.tensor([-10.,-1.,0.,1.,10.]);edge_out=launch(edge,torch.tensor([-10.,0.,1.,5.,10.]))
 n=1<<20;r=torch.randn(n);c=torch.randn(n)
 for _ in range(3):launch(r,c);F.relu6(r.square()+2*r+c*(1+r.abs()))
 def bench(fn):
  start=time.perf_counter()
  for _ in range(30):fn()
  return (time.perf_counter()-start)/30
 triton_s=bench(lambda:launch(r,c));torch_s=bench(lambda:F.relu6(r.square()+2*r+c*(1+r.abs())));parity=all(max(x["forward_max_abs"],x["receptive_grad_max_abs"],x["context_grad_max_abs"])<=2e-5 for x in cases)
 print(json.dumps({"gate1":"PASS","gate2":"PASS" if parity else "FAIL","cases":cases,"edge_output":edge_out.tolist(),"gate3_microbenchmark":{"elements":n,"triton_seconds":triton_s,"pytorch_seconds":torch_s,"speedup":torch_s/triton_s,"pass":torch_s/triton_s>=1.25}}))
if __name__=="__main__":main()
