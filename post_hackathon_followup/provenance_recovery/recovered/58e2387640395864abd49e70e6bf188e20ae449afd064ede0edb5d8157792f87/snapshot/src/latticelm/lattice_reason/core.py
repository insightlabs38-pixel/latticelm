"""Random-access symbolic worlds, solvers, verifiers, and surface realizers.

No label is inferred by a model: each family constructs explicit state and its
answer is recomputed by :func:`verify_example` from that state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Any, Callable

GENERATOR_VERSION = "lattice-reason-v1"
FAMILIES = ("entity_reference", "state_tracking", "spatial_relational",
            "temporal_causal", "boolean_logical", "arithmetic_quantity",
            "procedural_planning", "physical_closed_world")
MIXTURE_CYCLE = (FAMILIES[0],)*3 + (FAMILIES[1],)*3 + (FAMILIES[2],)*3 + \
                (FAMILIES[3],)*3 + (FAMILIES[4],)*3 + (FAMILIES[5],)*2 + \
                (FAMILIES[6],)*2 + (FAMILIES[7],)
TRAIN_TEMPLATES = {f: (f+"-train-a", f+"-train-b", f+"-train-c") for f in FAMILIES}
VALID_TEMPLATES = {f: (f+"-holdout-x", f+"-holdout-y") for f in FAMILIES}
TRAIN_NAMES = ("Ari", "Bela", "Ciro", "Dara", "Eli", "Fara", "Gino", "Hana")
VALID_NAMES = ("Ivo", "Juna", "Kato", "Lumi")
OBJECTS = ("token", "orb", "tile", "key", "ring", "cube", "card", "gem")
CONTAINERS = ("amber box", "blue bin", "cedar case", "dune tray", "emerald bag")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)+"\n").encode()


def stable_rng(seed: int, split: str, family: str, difficulty: int, index: int) -> random.Random:
    raw=f"{GENERATOR_VERSION}|{seed}|{split}|{family}|{difficulty}|{index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:16], "big"))


@dataclass(frozen=True)
class Example:
    global_id: str
    generator_version: str
    seed: int
    counter: int
    family: str
    subfamily: str
    template_family_id: str
    split: str
    difficulty: int
    symbolic_state: dict[str, Any]
    symbolic_state_hash: str
    verifier_result: str
    answer: str
    candidates: tuple[str, ...]
    hard_negative_strategy: str | None
    reasoning_trace: tuple[str, ...]
    rendered_text: str
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def _choice(rng: random.Random, values: tuple[str, ...], count: int) -> list[str]:
    return rng.sample(list(values), count)


def _entity(r: random.Random, d: int, split: str) -> tuple[dict, str, list[str], str, list[str]]:
    names=_choice(r, VALID_NAMES if split=="validation" else TRAIN_NAMES, min(4,2+(d>1)+(d>3)))
    objs=_choice(r, OBJECTS,len(names)); owner=dict(zip(objs,names)); focus=objs[0]; previous=owner[focus]
    transfers=[]
    for i in range(max(1,d)):
        obj=objs[i%len(objs)]; old=owner[obj]; new=names[(names.index(old)+1+i)%len(names)]
        transfers.append([obj,old,new]); owner[obj]=new
    answer=owner[focus]; wrong=previous if previous!=answer else names[(names.index(answer)+1)%len(names)]
    state={"names":names,"objects":objs,"initial_owners":dict(zip(objs,names)),"transfers":transfers,"query":focus}
    return state,answer,[wrong,names[(names.index(answer)+1)%len(names)]],"previous_state",[f"transfer {o}: {a} -> {b}" for o,a,b in transfers]


def _state(r: random.Random, d: int, split: str) -> tuple[dict,str,list[str],str,list[str]]:
    objs=_choice(r,OBJECTS,min(4,2+(d>2))); boxes=_choice(r,CONTAINERS,min(4,2+(d>1)))
    loc={o:boxes[i%len(boxes)] for i,o in enumerate(objs)}; initial=dict(loc); moves=[]; focus=objs[0]; previous=loc[focus]
    for i in range(max(1,d+1)):
        o=objs[i%len(objs)]; dst=boxes[(boxes.index(loc[o])+1+i)%len(boxes)]; previous=loc[focus] if o==focus else previous
        moves.append([o,dst]);loc[o]=dst
    answer=loc[focus]; wrong=previous if previous!=answer else boxes[(boxes.index(answer)+1)%len(boxes)]
    return {"objects":objs,"containers":boxes,"initial":initial,"moves":moves,"query":focus},answer,[wrong,boxes[(boxes.index(answer)+1)%len(boxes)]],"previous_state",[f"move {o} to {b}" for o,b in moves]


def _spatial(r: random.Random,d:int,split:str)->tuple[dict,str,list[str],str,list[str]]:
    pool=VALID_NAMES if split=="validation" else TRAIN_NAMES
    names=_choice(r,pool,min(len(pool),5,2+d)); order=names[:]
    # Coordinates make transitivity explicit and contradiction-free.
    coords={n:i for i,n in enumerate(order)}; left,right=order[0],order[-1]; answer="left"
    if r.randrange(2): left,right=right,left; answer="right"
    edges=[[order[i],"left_of",order[i+1]] for i in range(len(order)-1)]
    return {"coordinates":coords,"relations":edges,"query":[left,right]},answer,["right" if answer=="left" else "left","same"],"reversed_relation",[f"{a} left of {b}" for a,_,b in edges]


def _temporal(r: random.Random,d:int,split:str)->tuple[dict,str,list[str],str,list[str]]:
    events=[f"event-{x}" for x in _choice(r,OBJECTS,min(5,2+d))]; order=events[:]; a,b=order[0],order[-1]
    answer="before"; pairs=[[order[i],order[i+1]] for i in range(len(order)-1)]
    return {"events":events,"before_pairs":pairs,"query":[a,b]},answer,["after","unrelated"],"one_swap",[f"{x} before {y}" for x,y in pairs]


def _boolean(r: random.Random,d:int,split:str)->tuple[dict,str,list[str],str,list[str]]:
    values={"P":bool(r.randrange(2)),"Q":bool(r.randrange(2)),"R":bool(r.randrange(2))}
    ops=("and","or","xor","equiv"); op=ops[(d+r.randrange(len(ops)))%len(ops)]; a,b="P","Q"
    result={"and":values[a] and values[b],"or":values[a] or values[b],"xor":values[a]^values[b],"equiv":values[a]==values[b]}[op]
    answer="true" if result else "false"
    return {"values":values,"expression":[op,a,b]},answer,["false" if result else "true"],"negated_result",[f"{a}={values[a]}",f"{b}={values[b]}",f"apply {op}"]


def _arithmetic(r: random.Random,d:int,split:str)->tuple[dict,str,list[str],str,list[str]]:
    start=r.randint(3,20); value=start; ops=[]
    for i in range(max(1,d)):
        amount=r.randint(1,5); op="add" if (i+r.randrange(2))%2==0 or value<=amount else "subtract"
        value=value+amount if op=="add" else value-amount;ops.append([op,amount])
    wrong=value+(1 if r.randrange(2) else -1)
    return {"start":start,"operations":ops},str(value),[str(wrong),str(start)],"one_step_arithmetic_error",[f"start {start}"]+[f"{o} {n}" for o,n in ops]


def _planning(r: random.Random,d:int,split:str)->tuple[dict,str,list[str],str,list[str]]:
    locations=_choice(r,CONTAINERS,min(4,2+(d>2))); current=locations[0]; goal=locations[-1]
    actions=[]
    for i in range(len(locations)-1):actions.append({"name":f"go-{i}","pre":locations[i],"effect":locations[i+1]})
    valid=actions[0]["name"]; wrong=actions[-1]["name"] if len(actions)>1 else "wait"
    return {"initial":current,"goal":goal,"actions":actions,"query":"valid_next"},valid,[wrong,"wait"],"valid_action_wrong_time",[f"at {current}",f"need action with precondition {current}"]


def _physical(r: random.Random,d:int,split:str)->tuple[dict,str,list[str],str,list[str]]:
    obj=_choice(r,OBJECTS,1)[0]; props={"fragile":bool(r.randrange(2)),"flexible":bool(r.randrange(2)),"container":bool(r.randrange(2))}
    action=("bend" if r.randrange(2) else "store"); valid=props["flexible"] if action=="bend" else props["container"]
    answer="valid" if valid else "invalid"
    return {"object":obj,"properties":props,"synthetic_rules":{"bend":"flexible","store":"container"},"action":action},answer,["invalid" if valid else "valid"],"missing_required_property",[f"rule {action} requires {('flexible' if action=='bend' else 'container')}",f"property is {valid}"]


BUILDERS: dict[str,Callable] = dict(zip(FAMILIES,(_entity,_state,_spatial,_temporal,_boolean,_arithmetic,_planning,_physical)))


def solve(family: str, state: dict[str,Any]) -> str:
    if family=="entity_reference":
        owners=dict(state["initial_owners"])
        for obj,old,new in state["transfers"]:
            if owners[obj]!=old: raise ValueError("invalid transfer reference")
            owners[obj]=new
        return owners[state["query"]]
    if family=="state_tracking":
        loc=dict(state["initial"])
        for obj,dst in state["moves"]: loc[obj]=dst
        return loc[state["query"]]
    if family=="spatial_relational":
        a,b=state["query"]; ca,cb=state["coordinates"][a],state["coordinates"][b]
        return "left" if ca<cb else ("right" if ca>cb else "same")
    if family=="temporal_causal":
        graph={x:[] for x in state["events"]}
        for a,b in state["before_pairs"]: graph[a].append(b)
        a,b=state["query"]
        def reaches(x,y):
            seen=set(); stack=[x]
            while stack:
                q=stack.pop()
                if q==y:return True
                if q not in seen:seen.add(q);stack.extend(graph[q])
            return False
        return "before" if reaches(a,b) else ("after" if reaches(b,a) else "unrelated")
    if family=="boolean_logical":
        op,a,b=state["expression"]; x,y=state["values"][a],state["values"][b]
        return str({"and":x and y,"or":x or y,"xor":x^y,"equiv":x==y}[op]).lower()
    if family=="arithmetic_quantity":
        n=state["start"]
        for op,v in state["operations"]:n=n+v if op=="add" else n-v
        return str(n)
    if family=="procedural_planning":
        valid=[x["name"] for x in state["actions"] if x["pre"]==state["initial"]]
        if len(valid)!=1:raise ValueError("world lacks unique valid action")
        return valid[0]
    if family=="physical_closed_world":
        req=state["synthetic_rules"][state["action"]]
        return "valid" if state["properties"].get(req,False) else "invalid"
    raise ValueError(f"unknown family {family}")


def _render(family:str,state:dict,answer:str,candidates:list[str],template:str,r:random.Random,trace:list[str],include_trace:bool)->str:
    if family=="entity_reference": premise=f"Initial owners: {state['initial_owners']}. Transfers: {state['transfers']}. Who owns the {state['query']} now?"
    elif family=="state_tracking": premise=f"Locations begin as {state['initial']}. Apply moves {state['moves']}. Where is the {state['query']} afterward?"
    elif family=="spatial_relational": premise=f"Relations: {state['relations']}. Is {state['query'][0]} left or right of {state['query'][1]}?"
    elif family=="temporal_causal": premise=f"Ordering constraints: {state['before_pairs']}. Is {state['query'][0]} before or after {state['query'][1]}?"
    elif family=="boolean_logical": premise=f"Truth assignment: {state['values']}. Evaluate {state['expression'][1]} {state['expression'][0]} {state['expression'][2]}."
    elif family=="arithmetic_quantity": premise=f"Begin with {state['start']}. Apply {state['operations']} in order. What is the result?"
    elif family=="procedural_planning": premise=f"Initial place: {state['initial']}. Goal: {state['goal']}. Actions: {state['actions']}. Which action is valid next?"
    else: premise=f"Synthetic object {state['object']} has properties {state['properties']}. In this world the rules are {state['synthetic_rules']}. Is action {state['action']} valid?"
    opts=list(dict.fromkeys([answer]+candidates));r.shuffle(opts)
    if template.endswith(("train-b","holdout-y")): premise=f"Closed-world puzzle. {premise} Candidates: {' | '.join(opts)}"
    elif template.endswith("train-c"): premise=f"Premises -> {premise} Choose from [{', '.join(opts)}]."
    else: premise=f"Problem: {premise} Options: {', '.join(opts)}."
    suffix=(" Verified steps: "+"; ".join(trace)+"." if include_trace else "")
    return f"{premise}{suffix} Answer: {answer}\n"


def generate_example(index:int,seed:int=7319,split:str="train",family:str|None=None,difficulty:int|None=None,include_trace:bool=False)->Example:
    if index<0:raise ValueError("index must be nonnegative")
    if split not in {"train","validation"}:raise ValueError("invalid split")
    family=family or MIXTURE_CYCLE[index%len(MIXTURE_CYCLE)]
    if family not in FAMILIES:raise ValueError("invalid family")
    difficulty=difficulty or (1+(index//len(MIXTURE_CYCLE))%6)
    if not 1<=difficulty<=6:raise ValueError("difficulty outside 1..6")
    rng=stable_rng(seed,split,family,difficulty,index);templates=VALID_TEMPLATES[family] if split=="validation" else TRAIN_TEMPLATES[family]
    template=templates[(index//len(MIXTURE_CYCLE))%len(templates)]
    state,answer,negatives,strategy,trace=BUILDERS[family](rng,difficulty,split)
    solved=solve(family,state)
    if solved!=answer:raise RuntimeError("builder/solver contradiction")
    candidates=list(dict.fromkeys([answer]+[str(x) for x in negatives]))
    if len(candidates)<2 or candidates.count(answer)!=1 or any(x==answer for x in candidates[1:]):raise RuntimeError("invalid candidates")
    rendered=_render(family,state,answer,candidates[1:],template,rng,trace,include_trace)
    state_hash=hashlib.sha256(canonical_json(state)).hexdigest();gid=f"lr-{split[0]}-{index:012d}"
    return Example(gid,GENERATOR_VERSION,seed,index,family,family,template,split,difficulty,state,state_hash,"PASS",answer,tuple(candidates),strategy,tuple(trace),rendered)


def verify_example(example:Example)->bool:
    return (example.generator_version==GENERATOR_VERSION and solve(example.family,example.symbolic_state)==example.answer
            and hashlib.sha256(canonical_json(example.symbolic_state)).hexdigest()==example.symbolic_state_hash
            and len(example.candidates)==len(set(example.candidates)) and example.candidates.count(example.answer)==1
            and all(x!=example.answer for x in example.candidates if x!=example.answer)
            and example.template_family_id in (VALID_TEMPLATES if example.split=="validation" else TRAIN_TEMPLATES)[example.family])


def reconstruct(global_id:str,seed:int=7319,include_trace:bool=False)->Example:
    try:prefix,split,index=global_id.split("-")
    except ValueError as exc:raise ValueError("invalid global ID") from exc
    if prefix!="lr" or split not in {"t","v"}:raise ValueError("invalid global ID")
    return generate_example(int(index),seed,"train" if split=="t" else "validation",include_trace=include_trace)
