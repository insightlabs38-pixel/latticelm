"""Evaluate a native checkpoint on WikiText-103 raw test perplexity."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer


def bits_per_byte(negative_log_likelihood_nats: float, utf8_bytes: int) -> float:
 if utf8_bytes <= 0: raise ValueError('utf8_bytes must be positive')
 return negative_log_likelihood_nats / math.log(2) / utf8_bytes

def digest(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--tokenizer',required=True); p.add_argument('--output',required=True); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--threads',type=int,default=2); a=p.parse_args()
 torch.set_num_threads(a.threads); started=time.perf_counter(); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=LatticeConfig(**ck['config']); model=build_model(c); model.load_state_dict(ck['model']); model.eval(); tok=load_tokenizer(a.tokenizer)
 ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='test')
 groups=defaultdict(list); count=0; byte_count=0
 for text in ds['text']:
  byte_count += len(text.encode('utf-8'))
  ids=[1]+tok.encode(text)
  for start in range(0,len(ids)-1,c.context_length):
   end=min(len(ids)-1,start+c.context_length); left=max(0,end-c.context_length)
   x=ids[left:end]; y=ids[left+1:end+1]; score_from=start-left
   groups[len(x)].append((x,y,score_from)); count += len(y)-score_from
 nll=0.0
 with torch.inference_mode():
  for group in groups.values():
   for off in range(0,len(group),a.batch_size):
    chunk=group[off:off+a.batch_size]; x=torch.tensor([z[0] for z in chunk]); logits,_=model(x); lp=F.log_softmax(logits,-1)
    for row,(_,y,s) in enumerate(chunk): nll -= lp[row,s:].gather(-1,torch.tensor(y[s:])[:,None]).sum().item()
 result={'dataset':'Salesforce/wikitext','config':'wikitext-103-raw-v1','split':'test',
         'tokens_scored':count,'negative_log_likelihood':nll,'negative_log_likelihood_nats':nll,
         'perplexity':math.exp(nll/count),'utf8_bytes':byte_count,
         'negative_log_likelihood_bits':nll/math.log(2),
         'bits_per_byte':bits_per_byte(nll,byte_count),
         'byte_counting_policy':'sum len(text.encode(UTF-8)) for raw dataset rows; no inserted separators',
         'bos_policy':'prepend BOS per nonempty raw row as context only; BOS is never scored',
         'windowing_policy':'non-overlapping target windows; stride 128; maximum context 128',
         'stride':128,'context_length':c.context_length,
         'wall_seconds':time.perf_counter()-started,'checkpoint_sha256':digest(a.checkpoint),
         'tokenizer_sha256':digest(a.tokenizer),'pytorch_threads':a.threads}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
