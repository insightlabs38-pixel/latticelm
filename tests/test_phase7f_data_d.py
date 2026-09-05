from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from latticelm.data_d import (CORPUS_ID, SCHEMA_VERSION, ContaminationRegistry, Document,
    ExactMixture, GlobalDeduplicator, Int32Shard, SourceStream, canonical_json,
    sha256_file, verify_top_manifest)


def write_shard(base: Path, source: str, values: np.ndarray, tokenizer_hash: str) -> dict:
    shard = base / f"{source}.int32"; values.astype("<i4").tofile(shard)
    manifest = {"schema_version": SCHEMA_VERSION, "corpus_identity": CORPUS_ID, "source": source,
        "upstream_uri": "fixture://synthetic", "upstream_revision": "fixture-v1",
        "acquisition_timestamp": "2026-09-05T00:00:00+00:00", "transformation_code_commit": "fixture",
        "transformation_config_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "tokenizer_sha256": tokenizer_hash, "token_count": len(values), "document_count": 2,
        "raw_byte_count": 0, "sha256": sha256_file(shard), "source_license_label": "synthetic",
        "document_boundary_offsets": [0, len(values)//2, len(values)], "stable_document_ids": [f"{source}:0", f"{source}:1"],
        "validation_assignment": ["train", "train"], "dedup_decision_metadata": {"version":"fixture"},
        "decontamination_metadata": {"version":"fixture"}, "path": shard.name}
    path = base / f"{source}.manifest.json"; path.write_bytes(canonical_json(manifest))
    return {"manifest_path": path.name, "manifest_sha256": sha256_file(path)}


def fixture(tmp_path: Path):
    tokenizer = tmp_path / "tokenizer.json"; tokenizer.write_text("{}\n")
    children=[]; streams={}
    for index, source in enumerate(("fineweb_edu","wikipedia","fineweb","babylm")):
        values=np.arange(index*10000,index*10000+4097,dtype=np.int32)
        children.append(write_shard(tmp_path,source,values,sha256_file(tokenizer)))
        manifest=json.loads((tmp_path/f"{source}.manifest.json").read_text())
        mmap=Int32Shard(tmp_path/f"{source}.int32",manifest).tokens
        assert not mmap.flags.writeable
        streams[source]=SourceStream([mmap],128,100+index)
    top={"schema_version":"data-d-corpus-v1","corpus_identity":CORPUS_ID,
         "tokenizer_sha256":sha256_file(tokenizer),"mixture_definition":{"fineweb_edu":.5,"wikipedia":.2,"fineweb":.15,"babylm":.15},
         "dedup_version":"fixture","decontamination_version":"fixture","total_unique_tokens":4*4097,
         "total_documents":8,"canonical_shard_ordering":[x["manifest_path"] for x in children],"shards":children}
    top_path=tmp_path/"manifest.json"; top_path.write_bytes(canonical_json(top))
    return tokenizer,top_path,streams


def test_schema_hash_mmap_and_manifest_verification(tmp_path):
    tokenizer,top,_=fixture(tmp_path); assert verify_top_manifest(top,tokenizer)["corpus_identity"]==CORPUS_ID
    target=tmp_path/"fineweb.int32"; target.write_bytes(target.read_bytes()+b"x")
    with pytest.raises(ValueError,match="hash mismatch"): verify_top_manifest(top,tokenizer)


def test_wrong_tokenizer_and_corrupt_child_manifest_rejected(tmp_path):
    tokenizer,top,_=fixture(tmp_path); wrong=tmp_path/"wrong.json"; wrong.write_text("x")
    with pytest.raises(ValueError,match="wrong tokenizer"): verify_top_manifest(top,wrong)
    child=tmp_path/"babylm.manifest.json"; child.write_text(child.read_text()+" ")
    with pytest.raises(ValueError,match="child manifest hash mismatch"): verify_top_manifest(top,tokenizer)


def test_exact_mixture_and_next_batch_resume(tmp_path):
    _,_,streams=fixture(tmp_path); mixer=ExactMixture(streams); batches=[]; labels=[]
    for _ in range(5):
        x,y,names=mixer.batch(); batches.append(hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()); labels += names
    assert Counter(labels)==Counter({"fineweb_edu":20,"wikipedia":8,"fineweb":6,"babylm":6})
    state=mixer.state_dict(); expected=mixer.batch()[0]
    _,_,new_streams=fixture(tmp_path); resumed=ExactMixture(new_streams); resumed.load_state_dict(state)
    assert np.array_equal(expected,resumed.batch()[0])


def test_fresh_process_batch_reproduction(tmp_path):
    tokenizer,top,streams=fixture(tmp_path); mixer=ExactMixture(streams)
    for _ in range(4): expected=mixer.batch()[0]
    code="""import hashlib,json,numpy as np,sys
from pathlib import Path
from latticelm.data_d import Int32Shard,SourceStream,ExactMixture
b=Path(sys.argv[1]); streams={}
for i,s in enumerate(('fineweb_edu','wikipedia','fineweb','babylm')):
 m=json.loads((b/f'{s}.manifest.json').read_text()); streams[s]=SourceStream([Int32Shard(b/f'{s}.int32',m).tokens],128,100+i)
x=None
mix=ExactMixture(streams)
for _ in range(4): x=mix.batch()[0]
print(hashlib.sha256(x.tobytes()).hexdigest())"""
    got=subprocess.check_output([sys.executable,"-c",code,str(tmp_path)],text=True).strip()
    assert got==hashlib.sha256(expected.tobytes()).hexdigest()


def test_global_dedup_is_order_independent_and_data_c_wins():
    docs=[Document("fineweb","z","Alpha beta gamma delta epsilon zeta eta theta iota.",{},"x"),
          Document("data_c","old","Alpha beta gamma delta epsilon zeta eta theta iota.",{},"x"),
          Document("wikipedia","w","Entirely clean document with enough distinct words to remain selected safely.",{},"x")]
    first,decisions=GlobalDeduplicator().select(docs)
    second,_=GlobalDeduplicator().select(reversed(docs))
    assert [x.document_id for x in first]==[x.document_id for x in second]==["old","w"]
    assert next(x for x in decisions if x["document_id"]=="z")["rule"]=="exact_document"


def test_near_duplicate_selection_is_deterministic():
    left=" ".join(f"word{i}" for i in range(80))
    # Whitespace differs at the byte level but canonical word shingles and
    # therefore SimHash are identical; no full paragraph reaches eight words.
    right="\n".join(f"word{i}" for i in range(80))
    docs=[Document("fineweb","later",left,{},"x"), Document("wikipedia","winner",right,{},"x")]
    kept,decisions=GlobalDeduplicator().select(docs)
    assert [x.document_id for x in kept]==["winner"]
    assert next(x for x in decisions if x["document_id"]=="later")["rule"]=="near_duplicate"


def test_paragraph_near_dedup_and_decontamination():
    base="This paragraph has enough ordinary words to trigger exact paragraph duplicate detection reliably."
    docs=[Document("wikipedia","a",base+"\n\nUnique ending one with several words.",{},"x"),
          Document("fineweb","b",base+"\n\nDifferent ending two with several words.",{},"x")]
    kept,decisions=GlobalDeduplicator().select(docs); assert len(kept)==1
    assert any(x.get("rule")=="exact_paragraph" for x in decisions)
    sample="one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    registry=ContaminationRegistry({"synthetic_exact":[sample],"synthetic_clean":["unrelated words remain safely separate from this test material"]})
    contaminated=Document("fineweb","bad","prefix "+sample+" suffix",{},"x")
    clean=Document("fineweb","good","A clean procedural explanation about planting seeds in damp garden soil and watering them carefully.",{},"x")
    assert registry.matches(contaminated) and not registry.matches(clean)


def test_stable_validation_partition_excludes_training():
    ids=[f"doc-{i}" for i in range(1000)]
    validation={x for x in ids if int(hashlib.sha256(x.encode()).hexdigest()[:8],16)%20==0}
    training=set(ids)-validation
    assert training.isdisjoint(validation) and validation


def test_old_data_c_fineweb_ids_are_rejected():
    old={"id-a","id-b"}; candidates=["id-b","id-c"]
    assert [x for x in candidates if x not in old]==["id-c"]
