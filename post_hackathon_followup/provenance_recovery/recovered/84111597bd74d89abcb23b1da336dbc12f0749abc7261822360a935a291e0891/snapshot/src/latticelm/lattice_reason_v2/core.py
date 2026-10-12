"""Typed symbolic worlds with exact solvers, provenance and surface isolation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json, random
from typing import Any

GENERATOR_VERSION="lattice-reason-v2.0"
SCHEMA="lattice-reason-example-v2"
SKILLS=("entity_reference","state_tracking","spatial_relational","temporal_ordering",
 "causal_intervention","boolean_constraint","arithmetic_quantity","procedural_planning",
 "physical_affordance","set_count_comparison","graph_reachability","sequence_continuation",
 "counterfactual","necessary_sufficient","multi_constraint_selection")
SURFACES=("compact_symbolic","ordinary_prose","story","distractor_heavy","candidate_first",
 "question_first","direct_answer","multiple_choice","concise_rationale")
TRAIN_SURFACES=SURFACES[:7]; VALID_SURFACES=SURFACES[7:]
TRAIN_NAMES=("Ari","Bela","Ciro","Dara","Eli","Fara","Gino","Hana","Jori","Kira","Milo","Nara")
VALID_NAMES=("Orin","Pava","Rumi","Sela","Tovi","Vana")
OBJECTS=("amber key","brass gear","cerulean tile","dune orb","ebony card","flint cube","glass ring","hemp cord","ivory disk","jade pin","khaki bead","linen token")
ERRORS=("stale_state","skipped_transition","reversed_relation","incorrect_entity_binding","wrong_operation_order","one_arithmetic_step_wrong","correct_action_wrong_time","violated_precondition","n_minus_one_constraints","counterfactual_state_leakage","necessary_sufficient_confusion","irrelevant_fact_reliance")

def canonical(value:Any)->bytes:return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
def digest(value:Any)->str:return hashlib.sha256(canonical(value)).hexdigest()
def _rng(seed:int,split:str,index:int)->random.Random:return random.Random(int.from_bytes(hashlib.sha256(f"{GENERATOR_VERSION}|{seed}|{split}|{index}".encode()).digest()[:16],"big"))

@dataclass(frozen=True)
class World:
 entities:tuple[str,...]; initial:dict[str,int]; transitions:tuple[dict[str,Any],...]
 relations:tuple[tuple[str,str,str],...]; rules:tuple[dict[str,Any],...]; query:dict[str,Any]
 provenance:dict[str,Any]
@dataclass(frozen=True)
class Example:
 schema:str; generator_version:str; global_id:str; seed:int; counter:int; split:str
 world:World; world_state_hash:str; skills:tuple[str,...]; reasoning_depth:int
 composition_depth:int; branching_factor:int; distractor_count:int; reference_ambiguity:int
 counterfactual_distance:int; hard_negative_error_type:str; error_step:int|None
 surface_grammar:str; lexical_set:str; answer_format:str; prompt:str; answer:str
 candidates:tuple[str,...]; candidate_position:int; structured_steps:tuple[dict[str,Any],...]
 counterfactual_pair_id:str|None; counterfactual_of:str|None
 def to_dict(self):return asdict(self)

def apply_transition(state:dict[str,int],action:dict[str,Any])->dict[str,int]:
 out=dict(state);op=action["op"];entity=action["entity"]
 if op=="set":out[entity]=int(action["value"])
 elif op=="add":out[entity]=out[entity]+int(action["value"])
 elif op=="swap":other=action["other"];out[entity],out[other]=out[other],out[entity]
 else:raise ValueError("unknown transition")
 return out
def verify_transition(before,action,after):
 try:return apply_transition(before,action)==after
 except (KeyError,ValueError,TypeError):return False
def replay(world:World)->tuple[dict[str,int],tuple[dict[str,Any],...]]:
 state=dict(world.initial);steps=[]
 for i,action in enumerate(world.transitions):
  after=apply_transition(state,action);steps.append({"index":i,"before":state,"action":action,"after":after,"verified":verify_transition(state,action,after)});state=after
 return state,tuple(steps)
def _skill_witness(skill,rng,entities):
 """Return an exact primitive-specific witness and its numeric contribution."""
 if skill=="entity_reference":
  aliases={"it":entities[0],"they":entities[1]};return {"aliases":aliases,"reference":"it"},1
 if skill=="state_tracking":
  values=[rng.randint(1,5),rng.randint(6,10)];return {"states":values},values[-1]-values[0]
 if skill=="spatial_relational":
  coords={entities[0]:0,entities[1]:rng.choice((-2,2))};return {"coordinates":coords,"query":entities[:2]},1 if coords[entities[1]]>0 else -1
 if skill=="temporal_ordering":
  order=list(entities[:3]);return {"order":order,"query":[order[0],order[-1]]},len(order)-1
 if skill=="causal_intervention":
  base=rng.randint(1,4);effect=rng.randint(1,3);return {"base":base,"intervention_effect":effect,"do":True},effect
 if skill=="boolean_constraint":
  values=[bool(rng.randrange(2)),bool(rng.randrange(2))];return {"values":values,"op":"xor"},1 if values[0]^values[1] else -1
 if skill=="arithmetic_quantity":
  ops=[rng.randint(-3,3) or 1,rng.randint(-3,3) or -1];return {"start":0,"operations":ops},sum(ops)
 if skill=="procedural_planning":
  plan=[f"step-{i}" for i in range(1+rng.randrange(3))];return {"valid_plan":plan,"preconditions_verified":True},len(plan)
 if skill=="physical_affordance":
  flexible=bool(rng.randrange(2));return {"action":"bend","properties":{"flexible":flexible},"requires":"flexible"},1 if flexible else -1
 if skill=="set_count_comparison":
  left=sorted(rng.sample(range(8),3));right=sorted(rng.sample(range(8),2));return {"left":left,"right":right},len(set(left)-set(right))
 if skill=="graph_reachability":
  edges=[[entities[i],entities[i+1]] for i in range(len(entities)-1)];return {"edges":edges,"start":entities[0],"end":entities[-1]},1
 if skill=="sequence_continuation":
  start=rng.randint(0,5);step=rng.choice((1,2,3));seq=[start+i*step for i in range(4)];return {"sequence":seq,"next":seq[-1]+step},step
 if skill=="counterfactual":
  actual=rng.randint(1,5);alternate=actual+rng.choice((-2,2));return {"actual":actual,"counterfactual":alternate,"changed_fact":"intervention"},alternate-actual
 if skill=="necessary_sufficient":
  antecedent=bool(rng.randrange(2));consequent=bool(rng.randrange(2));return {"antecedent":antecedent,"consequent":consequent,"query":"necessary"},1 if (not consequent or antecedent) else -1
 if skill=="multi_constraint_selection":
  candidates={entities[0]:[True,True,True],entities[1]:[True,False,True]};return {"constraints":candidates,"required":3},1
 raise ValueError("unknown skill")
def _witness_delta(skill,w):
 if skill=="entity_reference":return 1 if w["aliases"][w["reference"]] else 0
 if skill=="state_tracking":return w["states"][-1]-w["states"][0]
 if skill=="spatial_relational":return 1 if w["coordinates"][w["query"][1]]>w["coordinates"][w["query"][0]] else -1
 if skill=="temporal_ordering":return w["order"].index(w["query"][1])-w["order"].index(w["query"][0])
 if skill=="causal_intervention":return w["intervention_effect"] if w["do"] else 0
 if skill=="boolean_constraint":return 1 if w["values"][0]^w["values"][1] else -1
 if skill=="arithmetic_quantity":return sum(w["operations"])
 if skill=="procedural_planning":return len(w["valid_plan"]) if w["preconditions_verified"] else 0
 if skill=="physical_affordance":return 1 if w["properties"].get(w["requires"],False) else -1
 if skill=="set_count_comparison":return len(set(w["left"])-set(w["right"]))
 if skill=="graph_reachability":return 1 if w["edges"] else -1
 if skill=="sequence_continuation":return w["next"]-w["sequence"][-1]
 if skill=="counterfactual":return w["counterfactual"]-w["actual"]
 if skill=="necessary_sufficient":return 1 if (not w["consequent"] or w["antecedent"]) else -1
 if skill=="multi_constraint_selection":return sum(all(v) for v in w["constraints"].values())
 raise ValueError("unknown skill")
def _reach(relations,start,end,label):
 graph={}
 for a,r,b in relations:
  if r==label:graph.setdefault(a,[]).append(b)
 seen=set();stack=[start]
 while stack:
  x=stack.pop()
  if x==end:return True
  if x not in seen:seen.add(x);stack.extend(graph.get(x,()))
 return False
def solve(world:World)->str:
 state,_=replay(world);q=world.query;kind=q["kind"]
 if kind=="value":return str(state[q["entity"]])
 if kind=="compare":return "greater" if state[q["left"]]>state[q["right"]] else ("less" if state[q["left"]]<state[q["right"]] else "equal")
 if kind=="reach":return "yes" if _reach(world.relations,q["start"],q["end"],q.get("relation","edge")) else "no"
 if kind=="relation":
  a,b=q["left"],q["right"]
  return "before" if _reach(world.relations,a,b,q["relation"]) else ("after" if _reach(world.relations,b,a,q["relation"]) else "unrelated")
 if kind=="eligible":
  eligible=[e for e in world.entities if all(state[e]>=int(rule["minimum"]) for rule in world.rules if rule["entity"]==e)]
  return eligible[0] if len(eligible)==1 else "none"
 if kind=="next_action":
  valid=[r["action"] for r in world.rules if r.get("precondition")==state[r["entity"]]]
  return valid[0] if len(valid)==1 else "none"
 raise ValueError("unknown query")

def _render(world,skills,surface,candidates,answer,rng,distractors):
 facts=[f"{k}={v}" for k,v in world.initial.items()]+[f"{x['op']}({x['entity']},{x.get('value',x.get('other'))})" for x in world.transitions]+[f"{a} {r} {b}" for a,r,b in world.relations]
 facts += [f"irrelevant-{i}: {OBJECTS[(i+3)%len(OBJECTS)]}" for i in range(distractors)]
 q=json.dumps(world.query,sort_keys=True);aux=world.provenance.get("auxiliary_queries",[]);q += (f"; auxiliary={json.dumps(aux,sort_keys=True)}" if aux else "");opts=list(candidates)
 if surface=="candidate_first":return f"Candidates: {' | '.join(opts)}. Facts: {'; '.join(facts)}. Query: {q}\nAnswer:"
 if surface=="question_first":return f"Query: {q}. Given: {'; '.join(facts)}. Options: {', '.join(opts)}\nAnswer:"
 if surface=="compact_symbolic":return f"F=[{' ; '.join(facts)}] Q={q} C={opts}\nA:"
 if surface=="story":return f"In a closed symbolic workshop, {'; then '.join(facts)}. Determine {q}. Choose {' / '.join(opts)}.\nAnswer:"
 if surface=="distractor_heavy":return f"Notes (some irrelevant): {'; '.join(facts)}. Task {q}. Candidates {opts}.\nAnswer:"
 if surface=="direct_answer":return f"{' ; '.join(facts)}. {q}\nDirect answer:"
 if surface=="concise_rationale":return f"Facts: {'; '.join(facts)}. Query: {q}. Give one verified step then answer from {opts}.\nReasoning:"
 return f"Facts: {'; '.join(facts)}. Question: {q}. Options: {', '.join(opts)}.\nAnswer:"

def generate_example(index:int,seed:int=260921,split:str="train",composition_depth:int|None=None,counterfactual=False)->Example:
 if index<0 or split not in {"train","validation"}:raise ValueError("invalid generation coordinate")
 rng=_rng(seed,split,index);depth=composition_depth or 1+index%4
 if not 1<=depth<=4:raise ValueError("composition depth outside 1..4")
 skills=tuple(SKILLS[(index*3+j)%len(SKILLS)] for j in range(depth));names=VALID_NAMES if split=="validation" else TRAIN_NAMES
 entities=tuple(rng.sample(names,3+min(2,depth)));initial={e:rng.randint(1,9) for e in entities};trans=[];witnesses=[];focus=entities[0]
 for skill in skills:
  witness,delta=_skill_witness(skill,rng,entities);witnesses.append({"skill":skill,"witness":witness,"delta":delta});trans.append({"op":"add","entity":focus,"value":delta,"source_skill":skill})
 for step in range(1+depth):
  entity=entities[(step+rng.randrange(len(entities)))%len(entities)];amount=rng.randint(1,4);trans.append({"op":"add","entity":entity,"value":amount if step%2==0 else -amount})
 relations=tuple((entities[i],label,entities[i+1]) for label in ("edge","before") for i in range(len(entities)-1));rules=()
 mode=index%5
 if depth>1:q={"kind":"value","entity":focus}
 elif mode==0:q={"kind":"value","entity":entities[index%len(entities)]}
 elif mode==1:q={"kind":"compare","left":entities[0],"right":entities[-1]}
 elif mode==2:
  a,b=(entities[0],entities[-1]) if rng.randrange(2) else (entities[-1],entities[0]);q={"kind":"reach","start":a,"end":b,"relation":"edge"}
 elif mode==3:
  orientation=rng.randrange(3)
  if orientation==0:a,b=entities[0],entities[-1]
  elif orientation==1:a,b=entities[-1],entities[0]
  else:
   # Remove the query relation to create a truly unrelated exact case.
   a,b=entities[0],entities[-1];relations=tuple(x for x in relations if x[1]!="before")
  q={"kind":"relation","left":a,"right":b,"relation":"before"}
 else:
  # Exactly one valid action; names and rule order are shuffled to prevent position cues.
  chosen=entities[rng.randrange(len(entities))];rules=tuple({"entity":e,"action":f"act-{rng.randrange(100,999)}-{e}","precondition":initial[e]+sum(t["value"] for t in trans if t["entity"]==e)+(0 if e==chosen else 1)} for e in rng.sample(list(entities),len(entities)));q={"kind":"next_action"}
 provenance={"seed":seed,"split":split,"index":index,"lexical_partition":"heldout" if split=="validation" else "train","world_seed":digest([seed,split,index]),"skill_witnesses":witnesses}
 if depth>=3:
  state=dict(initial)
  for action in trans:state=apply_transition(state,action)
  provenance["auxiliary_queries"]=[{"query":{"kind":"value","entity":entities[1]},"answer":str(state[entities[1]])},{"query":{"kind":"compare","left":entities[0],"right":entities[1]},"answer":"greater" if state[entities[0]]>state[entities[1]] else ("less" if state[entities[0]]<state[entities[1]] else "equal")}]
 world=World(entities,initial,tuple(trans),relations,rules,q,provenance);answer=solve(world);state,steps=replay(world)
 wrong=[]
 if answer.lstrip("-").isdigit():wrong=[str(int(answer)+1),str(int(answer)-1),str(initial.get(q.get("entity",""),0))]
 elif answer in ("yes","no"):wrong=["no" if answer=="yes" else "yes","unknown"]
 elif answer in ("greater","less","equal"):wrong=[x for x in ("greater","less","equal") if x!=answer]
 elif answer in ("before","after","unrelated"):wrong=[x for x in ("before","after","unrelated") if x!=answer]
 else:wrong=[r["action"] for r in rules if r["action"]!=answer]+["wait"]
 candidates=list(dict.fromkeys([answer,*wrong]))[:4];rng.shuffle(candidates);surface=(VALID_SURFACES if split=="validation" else TRAIN_SURFACES)[(index//3)%len(VALID_SURFACES if split=="validation" else TRAIN_SURFACES)];distractors=(index//7)%6
 gid=f"lrv2-{split[0]}-{index:016d}";pair=f"cf-{seed}-{index//2}" if counterfactual else None
 prompt=_render(world,skills,surface,tuple(candidates),answer,rng,distractors)
 return Example(SCHEMA,GENERATOR_VERSION,gid,seed,index,split,world,digest(asdict(world)),skills,len(steps),depth,len(candidates),distractors,max(0,len(entities)-3),1 if counterfactual else 0,ERRORS[index%len(ERRORS)],min(len(steps)-1,index%max(1,len(steps))),surface,"heldout-v1" if split=="validation" else "train-v1","direct" if surface=="direct_answer" else "candidate",prompt,answer,tuple(candidates),candidates.index(answer),steps,pair,None)
def verify_candidate(example,candidate):return candidate==solve(example.world)
def verify_example(example):
 try:
  witnesses=example.world.provenance["skill_witnesses"];skill_actions=[x for x in example.world.transitions if x.get("source_skill")]
  composed=(tuple(x["skill"] for x in witnesses)==example.skills and len(skill_actions)==len(example.skills) and all(_witness_delta(x["skill"],x["witness"])==x["delta"]==a["value"] and a["source_skill"]==x["skill"] for x,a in zip(witnesses,skill_actions)))
  auxiliary=all(solve(World(example.world.entities,example.world.initial,example.world.transitions,example.world.relations,example.world.rules,x["query"],example.world.provenance))==x["answer"] for x in example.world.provenance.get("auxiliary_queries",[]))
  return composed and auxiliary and example.schema==SCHEMA and example.generator_version==GENERATOR_VERSION and digest(asdict(example.world))==example.world_state_hash and solve(example.world)==example.answer and example.candidates.count(example.answer)==1 and all(s["verified"] and verify_transition(s["before"],s["action"],s["after"]) for s in example.structured_steps) and ((example.surface_grammar in VALID_SURFACES)==(example.split=="validation"))
 except (KeyError,ValueError,TypeError):return False
def reconstruct(global_id,seed=260921):
 prefix,split,index=global_id.split("-")
 if prefix!="lrv2" or split not in {"t","v"}:raise ValueError("invalid v2 ID")
 return generate_example(int(index),seed,"train" if split=="t" else "validation")
def counterfactual_pair(index,seed=260921,split="train"):
 base=generate_example(index,seed,split,counterfactual=True);w=base.world;initial=dict(w.initial);key=w.query.get("entity") or w.query.get("left") or w.entities[0];initial[key]+=7
 changed_provenance={k:v for k,v in w.provenance.items() if k!="auxiliary_queries"};changed_provenance["counterfactual_change"]={key:7};changed=World(w.entities,initial,w.transitions,w.relations,w.rules,w.query,changed_provenance);answer=solve(changed)
 if answer==base.answer:
  # Relevant transition inversion guarantees a value-query answer change.
  q={"kind":"value","entity":key};changed=World(w.entities,initial,w.transitions,w.relations,w.rules,q,changed_provenance);answer=solve(changed)
 cf=Example(**{**base.__dict__,"global_id":base.global_id+"-cf","world":changed,"world_state_hash":digest(asdict(changed)),"answer":answer,"candidates":tuple(dict.fromkeys((answer,*base.candidates))),"candidate_position":0,"structured_steps":replay(changed)[1],"counterfactual_of":base.global_id})
 return base,cf
