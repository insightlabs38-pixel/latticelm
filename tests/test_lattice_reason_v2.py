from dataclasses import replace
from latticelm.lattice_reason_v2 import generate_example,reconstruct,verify_candidate,verify_example,verify_transition
from latticelm.lattice_reason_v2.core import TRAIN_NAMES,VALID_NAMES,TRAIN_SURFACES,VALID_SURFACES,counterfactual_pair
from latticelm.lattice_reason_v2.audit import contamination_hits,shortcut_audit
from latticelm.lattice_reason_v2.natural import generate_from_tokens,verify as verify_natural
def test_deterministic_random_access_and_disjointness():
 a=generate_example(91);b=generate_example(91);v=generate_example(91,split="validation");assert a==b and reconstruct(a.global_id)==a;assert set(a.world.entities)<=set(TRAIN_NAMES) and set(v.world.entities)<=set(VALID_NAMES);assert a.surface_grammar in TRAIN_SURFACES and v.surface_grammar in VALID_SURFACES
def test_solver_candidate_and_process_verification():
 for i in range(100):
  x=generate_example(i,composition_depth=1+i%4);assert verify_example(x) and verify_candidate(x,x.answer);assert not verify_candidate(x,next(c for c in x.candidates if c!=x.answer));assert all(verify_transition(s["before"],s["action"],s["after"]) for s in x.structured_steps)
def test_counterfactual_changes_answer_and_world_hash():
 for i in range(10):
  a,b=counterfactual_pair(i);assert a.answer!=b.answer and a.world_state_hash!=b.world_state_hash and verify_example(a) and verify_example(b)
def test_hard_negatives_are_wrong_and_typed():
 for i in range(100):
  x=generate_example(i);assert x.hard_negative_error_type and x.error_step is not None and all((c==x.answer)==verify_candidate(x,c) for c in x.candidates)
def test_shortcut_audit_passes_balanced_v2_and_catches_trivial():
 rows=[generate_example(i,split="validation") for i in range(2000)];assert shortcut_audit(rows,.16)["status"]=="PASS";trivial=[replace(x,answer="yes",candidates=("yes","no"),candidate_position=0) for x in rows[:100]];assert shortcut_audit(trivial,.1)["status"]=="FAIL"
def test_contamination_and_natural_provenance():
 phrase="these thirteen unique benchmark words must never enter generated training examples at all today";assert contamination_hits("prefix "+phrase+" suffix",{"gibc":[phrase]})==["gibc"];tokens=list(range(1000));x=generate_from_tokens(tokens,3);assert verify_natural(x,tokens) and all(len(c)==len(x.answer_ids) for c in x.candidates)
def test_context_metadata_and_structural_dimensions():
 x=generate_example(4,composition_depth=4);assert x.composition_depth==4 and x.reasoning_depth>=4 and x.branching_factor==len(x.candidates) and len(x.prompt)>0
