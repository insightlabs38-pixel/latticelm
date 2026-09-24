"""Exact provenance transformations over certified DATA-D-v4 token streams."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,random

@dataclass(frozen=True)
class NaturalExample:
 global_id:str; transformation:str; source_id:str; source_offset:int; prompt_ids:tuple[int,...]
 answer_ids:tuple[int,...]; candidates:tuple[tuple[int,...],...]; provenance_hash:str
 negative_offsets:tuple[int,...]=()
 def to_dict(self):return asdict(self)
TRANSFORMATIONS=("source_contiguous_next_span","boundary_shifted_continuation","locally_perturbed_order","exact_token_corruption")
def generate_from_tokens(tokens,index,source_id="data-d-v4",seed=260921,prompt=96,answer=16,transformation=None):
 """Mechanically labeled DATA-D tasks with exact source token targets."""
 if len(tokens)<prompt+answer*4:raise ValueError("source too short")
 transformation=transformation or TRANSFORMATIONS[index%len(TRANSFORMATIONS)]
 if transformation not in TRANSFORMATIONS:raise ValueError(transformation)
 rng=random.Random(int.from_bytes(hashlib.sha256(f"natural-v2|{seed}|{source_id}|{index}".encode()).digest()[:16],"big"));limit=len(tokens)-prompt-answer-3;start=rng.randrange(3,limit);p=tuple(int(x) for x in tokens[start:start+prompt]);correct=tuple(int(x) for x in tokens[start+prompt:start+prompt+answer]);offsets=[];wrong=[]
 if transformation=="source_contiguous_next_span":
  while len(offsets)<3:
   x=rng.randrange(limit)
   if abs(x-(start+prompt))>=answer and x not in offsets:offsets.append(x)
  wrong=[tuple(int(y) for y in tokens[x:x+answer]) for x in offsets]
 elif transformation=="boundary_shifted_continuation":
  offsets=[start+prompt+d for d in (-2,-1,1)];wrong=[tuple(int(y) for y in tokens[x:x+answer]) for x in offsets]
 elif transformation=="locally_perturbed_order":
  chunks=[correct[i:i+max(1,answer//4)] for i in range(0,answer,max(1,answer//4))]
  wrong=[tuple(y for j in order for y in chunks[j]) for order in ((1,0,2,3),(0,2,1,3),(3,1,2,0))]
 else:
  for position in (answer//4,answer//2,3*answer//4):
   candidate=list(correct);candidate[position]=int(tokens[start+prompt+answer+position]);wrong.append(tuple(candidate))
 candidates=list(dict.fromkeys([correct,*wrong]));rng.shuffle(candidates);prov={"source_id":source_id,"start":start,"prompt":prompt,"answer":answer,"transformation":transformation,"candidates":sorted(candidates),"negative_offsets":offsets};digest=hashlib.sha256(repr(sorted(prov.items())).encode()).hexdigest()
 return NaturalExample(f"lrv2-natural-{index:016d}",transformation,source_id,start,p,correct,tuple(candidates),digest,tuple(offsets))
def verify(example,tokens):
 start=example.source_offset;p=len(example.prompt_ids);a=len(example.answer_ids)
 if not (tuple(int(x) for x in tokens[start:start+p])==example.prompt_ids and tuple(int(x) for x in tokens[start+p:start+p+a])==example.answer_ids and example.answer_ids in example.candidates and all(len(x)==a for x in example.candidates)):return False
 if example.transformation in ("source_contiguous_next_span","boundary_shifted_continuation"):
  if any(tuple(int(y) for y in tokens[x:x+a]) not in example.candidates for x in example.negative_offsets):return False
 elif example.transformation=="locally_perturbed_order":
  if any(sorted(c)!=sorted(example.answer_ids) for c in example.candidates):return False
 elif example.transformation=="exact_token_corruption":
  if any(sum(x!=y for x,y in zip(c,example.answer_ids))>1 for c in example.candidates):return False
 else:return False
 prov={"source_id":example.source_id,"start":start,"prompt":p,"answer":a,"transformation":example.transformation,"candidates":sorted(example.candidates),"negative_offsets":list(example.negative_offsets)}
 return hashlib.sha256(repr(sorted(prov.items())).encode()).hexdigest()==example.provenance_hash
