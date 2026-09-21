"""Shortcut and contamination certification for materialized v2 examples."""
from collections import Counter,defaultdict
import math,re
from .core import verify_example

def _accuracy(rows,key):
 groups=defaultdict(Counter)
 for row in rows:groups[str(key(row))][row.answer]+=1
 return sum(max(c.values()) for c in groups.values())/len(rows)
def shortcut_audit(rows,tolerance=.10):
 rows=list(rows)
 if not rows:raise ValueError("empty audit")
 answers=Counter(x.answer for x in rows);chance=sum(1/len(x.candidates) for x in rows)/len(rows)
 metrics={"examples":len(rows),"answer_distribution":dict(answers),"chance":chance,
  "majority_answer_baseline":max(answers.values())/len(rows),
  "candidate_position_baseline":max(Counter(x.candidate_position for x in rows).values())/len(rows),
  "surface_to_answer_predictability":_accuracy(rows,lambda x:x.surface_grammar),
  "family_to_answer_predictability":_accuracy(rows,lambda x:"+".join(x.skills)),
  "difficulty_to_answer_predictability":_accuracy(rows,lambda x:(x.composition_depth,x.reasoning_depth,x.distractor_count))}
 metrics["status"]="PASS" if all(metrics[k]<=chance+tolerance for k in ("majority_answer_baseline","candidate_position_baseline","surface_to_answer_predictability","family_to_answer_predictability","difficulty_to_answer_predictability")) else "FAIL"
 return metrics
def contamination_hits(text,registry,ngram=13):
 words=re.findall(r"[a-z0-9]+",text.lower());joined=" ".join(words);hits=[]
 if registry.get("schema")=="lattice-reason-v2-contamination-registry-v1":
  exact=set(registry["exact"]);grams=set(registry["grams"]);own={int.from_bytes(__import__("hashlib").blake2b(" ".join(words[i:i+ngram]).encode(),digest_size=8).digest(),"big") for i in range(max(0,len(words)-ngram+1))}
  return ["certified-benchmark-registry"] if joined in exact or own.intersection(grams) else []
 for name,items in registry.items():
  for item in items:
   ref=re.findall(r"[a-z0-9]+",item.lower())
   if (len(ref)>=8 and " ".join(ref) in joined) or any(tuple(words[i:i+ngram])==tuple(ref[j:j+ngram]) for i in range(max(0,len(words)-ngram+1)) for j in range(max(0,len(ref)-ngram+1))):hits.append(name);break
 return sorted(set(hits))
def compile_registry(rows,ngram=13):
 grams=set();exact=set()
 for text in rows:
  words=re.findall(r"[a-z0-9]+",text.lower())
  if len(words)>=8:exact.add(" ".join(words))
  for i in range(max(0,len(words)-ngram+1)):grams.add(int.from_bytes(__import__("hashlib").blake2b(" ".join(words[i:i+ngram]).encode(),digest_size=8).digest(),"big"))
 return {"schema":"lattice-reason-v2-contamination-registry-v1","ngram":ngram,"exact":sorted(exact),"grams":sorted(grams),"source":"benchmark train/validation/test plus GIBC registry"}
def certify(rows,registry=None):
 rows=list(rows);bad=[x.global_id for x in rows if not verify_example(x)];contaminated={x.global_id:contamination_hits(x.prompt,registry or {}) for x in rows};contaminated={k:v for k,v in contaminated.items() if v};audit=shortcut_audit(rows)
 return {"verification_failures":bad,"contamination":contaminated,"shortcut_audit":audit,"status":"PASS" if not bad and not contaminated and audit["status"]=="PASS" else "FAIL"}
