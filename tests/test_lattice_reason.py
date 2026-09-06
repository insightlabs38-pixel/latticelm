from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import pytest

from latticelm.lattice_reason.core import (FAMILIES,MIXTURE_CYCLE,TRAIN_TEMPLATES,VALID_TEMPLATES,
    generate_example,reconstruct,solve,verify_example)
from latticelm.lattice_reason.dataset import (build_canonical,build_validation,contamination_matches,
    deadline_allows,disk_allows,select_adaptive_target,sha256_file,verify_manifest)

ROOT=Path(__file__).resolve().parents[1];TOKENIZER=ROOT/"artifacts/tokenizers/babylm_2026_4k.json"

@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("difficulty",range(1,7))
def test_every_family_difficulty_is_verified(family,difficulty):
    for i in range(8):
        ex=generate_example(i+100*difficulty,17,"train",family,difficulty,include_trace=True)
        assert verify_example(ex);assert solve(family,ex.symbolic_state)==ex.answer
        assert ex.answer in ex.candidates and ex.candidates.count(ex.answer)==1
        assert len(ex.candidates)==len(set(ex.candidates));assert ex.hard_negative_strategy

def test_deterministic_rendering_regeneration_and_ids():
    examples=[generate_example(i,91) for i in range(500)]
    assert examples==[generate_example(i,91) for i in range(500)]
    assert len({x.global_id for x in examples})==len(examples)
    assert all(reconstruct(x.global_id,91)==x for x in examples)

def test_template_and_vocabulary_holdouts():
    assert not set().union(*TRAIN_TEMPLATES.values()) & set().union(*VALID_TEMPLATES.values())
    train=[generate_example(i,3,"train") for i in range(500)]
    valid=[generate_example(i,3,"validation") for i in range(500)]
    assert not {x.template_family_id for x in train}&{x.template_family_id for x in valid}
    assert not set().union(*(set(x.symbolic_state.get("names",[])) for x in train)) & set().union(*(set(x.symbolic_state.get("names",[])) for x in valid))

def test_family_and_difficulty_scheduler():
    rows=[generate_example(i) for i in range(1200)];counts=Counter(x.family for x in rows)
    expected=Counter(MIXTURE_CYCLE)
    assert all(counts[k]==expected[k]*60 for k in FAMILIES)
    assert Counter(x.difficulty for x in rows)=={i:200 for i in range(1,7)}

def test_contamination_rejection():
    phrase="one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    assert contamination_matches("prefix "+phrase+" suffix",{"arc_easy":[phrase]})==["arc_easy"]
    assert contamination_matches("entirely unrelated generated statement",{"arc_easy":[phrase]})==[]

def test_guards_and_adaptive_selection(tmp_path):
    assert select_adaptive_target(100)==1_000_000_000
    assert select_adaptive_target(4000)==500_000_000
    assert select_adaptive_target(20_000)==100_000_000
    assert deadline_allows(0,10,1000,20,30);assert not deadline_allows(0,950,1000,20,30)
    assert disk_allows(tmp_path,1,0);assert not disk_allows(tmp_path,10**30,0)

def test_shards_views_resume_roundtrip_and_corruption(tmp_path):
    # Canonical APIs require public milestone names; a large shard keeps this fixture quick.
    out=tmp_path/"pool";result=build_canonical(out,TOKENIZER,1_000_000,5,1_000_000,"test",300,600,min_free_bytes=0)
    assert result["token_count"]==1_000_000 and result["views"][0]["tokens"]==1_000_000
    view=out/result["views"][0]["path"];assert verify_manifest(view,TOKENIZER)["exact_tokens"]==1_000_000
    shard=out/result["shards"][0]["path"];arr=np.memmap(shard,mode="r",dtype="<i4");assert len(arr)==1_000_000
    again=build_canonical(out,TOKENIZER,1_000_000,5,1_000_000,"test",300,600,min_free_bytes=0)
    assert again["token_count"]==result["token_count"]
    data=json.loads(view.read_text());data["tokenizer_sha256"]="0"*64;view.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="tokenizer"):verify_manifest(view,TOKENIZER)

def test_incomplete_shard_rollback(tmp_path):
    out=tmp_path/"pool";out.mkdir();(out/"orphan.partial").write_bytes(b"bad")
    build_canonical(out,TOKENIZER,1_000_000,7,1_000_000,"test",300,600,min_free_bytes=0)
    assert not list(out.glob("*.partial"))

def test_fixed_validation_suite(tmp_path):
    manifest=build_validation(tmp_path,TOKENIZER,2,44,{"dummy":["benchmark phrase that cannot occur in these synthetic worlds at all"]})
    assert manifest["examples"]==len(FAMILIES)*6*2 and manifest["verification"]=="PASS"
    assert sha256_file(tmp_path/manifest["data_path"])==manifest["data_sha256"]
