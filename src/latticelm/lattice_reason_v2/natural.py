"""Exact provenance transformations over certified DATA-D-v4 token streams."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,random

@dataclass(frozen=True)
class NaturalExample:
 global_id:str; transformation:str; source_id:str; source_offset:int; prompt_ids:tuple[int,...]
 answer_ids:tuple[int,...]; candidates:tuple[tuple[int,...],...]; provenance_hash:str
 def to_dict(self):return asdict(self)
def generate_from_tokens(tokens,index,source_id="data-d-v4",seed=260921,prompt=96,answer=16):
 """Random-access next-span task; wrong spans are same-source and length matched."""
 if len(tokens)<prompt+answer*4:raise ValueError("source too short")
 rng=random.Random(int.from_bytes(hashlib.sha256(f"natural-v2|{seed}|{source_id}|{index}".encode()).digest()[:16],"big"));limit=len(tokens)-prompt-answer;start=rng.randrange(limit);p=tuple(int(x) for x in tokens[start:start+prompt]);correct=tuple(int(x) for x in tokens[start+prompt:start+prompt+answer]);offsets=[]
 while len(offsets)<3:
  x=rng.randrange(limit)
  if abs(x-(start+prompt))>=answer and x not in offsets:offsets.append(x)
 candidates=[correct]+[tuple(int(y) for y in tokens[x:x+answer]) for x in offsets];rng.shuffle(candidates);prov={"source_id":source_id,"start":start,"prompt":prompt,"answer":answer,"distractor_offsets":offsets};digest=hashlib.sha256(repr(sorted(prov.items())).encode()).hexdigest()
 return NaturalExample(f"lrv2-natural-{index:016d}","source_contiguous_next_span",source_id,start,p,correct,tuple(candidates),digest)
def verify(example,tokens):
 start=example.source_offset;p=len(example.prompt_ids);a=len(example.answer_ids)
 return tuple(int(x) for x in tokens[start:start+p])==example.prompt_ids and tuple(int(x) for x in tokens[start+p:start+p+a])==example.answer_ids and example.answer_ids in example.candidates and all(len(x)==a for x in example.candidates)
