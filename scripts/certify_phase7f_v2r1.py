"""Independent acceptance verifier for DATA-D-BROAD-v2r1."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from latticelm.config import LatticeConfig
from latticelm.data_d import (CORPUS_V2R1_ID, ExactMixtureV2, Int32Shard, SourceStream,
    document_hash, hamming, paragraph_hashes, sha256_file, simhash64,
    validation_assignment, verify_top_manifest)
from latticelm.model import build_model

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "artifacts/tokenizers/babylm_2026_4k.json"
SOURCES = ("fineweb_edu", "wikipedia", "fineweb")


def arrays(base: Path, children: list[dict], split: str) -> tuple[dict[str, list[np.ndarray]], set[str]]:
    output = {source: [] for source in SOURCES}; ids: set[str] = set()
    for child in children:
        path = base / child["manifest_path"]
        if sha256_file(path) != child["manifest_sha256"]: raise ValueError(f"{split} child manifest hash mismatch")
        manifest = json.loads(path.read_text())
        if manifest["split"] != split: raise ValueError(f"wrong {split} split label")
        if manifest["tokenizer_sha256"] != sha256_file(TOKENIZER): raise ValueError("wrong tokenizer in child")
        doc_ids = manifest["stable_document_ids"]
        if any(validation_assignment(value) != split for value in doc_ids): raise ValueError("validation assignment mismatch")
        if ids.intersection(doc_ids): raise ValueError(f"duplicate {split} document ID")
        ids.update(doc_ids)
        shard = Int32Shard(base / manifest["path"], manifest)
        if shard.tokens.flags.writeable: raise ValueError("mmap is writable")
        output[manifest["source"]].append(shard.tokens)
    return output, ids


def exact_resume_probe(base: Path, manifest: dict) -> dict:
    train_arrays, _ = arrays(base, manifest["shards"], "train")

    def setup():
        random.seed(90210); torch.manual_seed(90210)
        streams = {source: SourceStream(train_arrays[source], 16, 410 + i) for i, source in enumerate(SOURCES)}
        mixer = ExactMixtureV2(streams)
        config = LatticeConfig(vocab_size=4096, d_model=32, n_layers=1, n_heads=4, n_kv_heads=1,
            ffn_hidden=64, context_length=16, batch_size=8, architecture="co4_causal",
            tie_embeddings=False, dropout=0.0, learning_rate=3e-4, seed=90210)
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return model, optimizer, scheduler, mixer

    def step(parts):
        model, optimizer, scheduler, mixer = parts
        x, y, _ = mixer.batch(); random.random()
        xt, yt = torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
        _, loss = model(xt, yt); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
        return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(), loss.detach().clone()

    original = setup()
    for _ in range(3): step(original)
    model, optimizer, scheduler, mixer = original
    checkpoint = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
        "python_rng_state":random.getstate(),"torch_rng_state":torch.get_rng_state(),"data_source_selector_state":mixer.state_dict(),
        "step":3,"tokens_seen":384,"manifest_sha256":sha256_file(base/"manifest.json")}
    payload = io.BytesIO(); torch.save(checkpoint, payload)
    expected_batch, expected_loss = step(original)
    expected_model = {key:value.detach().clone() for key,value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict(); expected_scheduler = scheduler.state_dict()
    expected_python, expected_torch = random.getstate(), torch.get_rng_state().clone()

    resumed = setup(); state = torch.load(io.BytesIO(payload.getvalue()), map_location="cpu", weights_only=False)
    model2, optimizer2, scheduler2, mixer2 = resumed
    if state["manifest_sha256"] != sha256_file(base/"manifest.json"): raise ValueError("resume manifest identity mismatch")
    model2.load_state_dict(state["model"]); optimizer2.load_state_dict(state["optimizer"]); scheduler2.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng_state"]); torch.set_rng_state(state["torch_rng_state"]); mixer2.load_state_dict(state["data_source_selector_state"])
    got_batch, got_loss = step(resumed)
    exact_model = all(torch.equal(value, model2.state_dict()[key]) for key,value in expected_model.items())
    if got_batch != expected_batch or not torch.equal(got_loss, expected_loss) or not exact_model:
        raise ValueError("full exact-resume equivalence failed")
    def same(left,right):
        if torch.is_tensor(left):return torch.equal(left,right)
        if isinstance(left,dict):return left.keys()==right.keys() and all(same(left[key],right[key]) for key in left)
        if isinstance(left,(list,tuple)):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
        return left==right
    if not same(expected_optimizer,optimizer2.state_dict()) or not same(expected_scheduler,scheduler2.state_dict()):
        raise ValueError("optimizer/scheduler resume mismatch")
    if random.getstate()!=expected_python or not torch.equal(torch.get_rng_state(),expected_torch):raise ValueError("RNG resume mismatch")
    return {"status":"PASS","interruption_step":3,"resumed_step":4,"tokens_seen":512,"next_batch_sha256":got_batch,"policy":"bit-exact CPU eager"}


def verify_audit(base: Path, manifest: dict, train_ids: set[str], val_ids: set[str]) -> dict:
    cache_path=base/manifest["audit_files"]["normalized_source_cache"]["path"]
    audit_path=base/manifest["audit_files"]["document_audit"]["path"]
    if sha256_file(cache_path)!=manifest["audit_files"]["normalized_source_cache"]["sha256"]:raise ValueError("source cache hash mismatch")
    if sha256_file(audit_path)!=manifest["audit_files"]["document_audit"]["sha256"]:raise ValueError("audit hash mismatch")
    cache={row["document_id"]:row for row in map(json.loads,cache_path.open())}
    retained_exact={};retained_paragraph={};retained_simhash={};bands=[{} for _ in range(4)];counts=Counter()
    for row in map(json.loads,audit_path.open()):
        source=cache.get(row["document_id"])
        if source is None:raise ValueError("audit row lacks source text")
        text=source["normalized_text"]
        if document_hash(text)!=row["normalized_text_sha256"] or list(paragraph_hashes(text))!=row["paragraph_sha256"] or f"{simhash64(text):016x}"!=row["simhash64"]:
            raise ValueError("audit fingerprint recomputation mismatch")
        counts[row["decision"]]+=1
        if row["decision"]=="retained":
            expected = train_ids if row["reason"]=="train" else val_ids
            if row["document_id"] not in expected:raise ValueError("retained audit/shard mismatch")
            exact=row["normalized_text_sha256"]
            if exact in retained_exact:raise ValueError("retained exact duplicate")
            for value in row["paragraph_sha256"]:
                if value in retained_paragraph:raise ValueError("retained paragraph duplicate")
            fingerprint=int(row["simhash64"],16)
            candidates={}
            for band in range(4):
                key=(fingerprint>>(16*band))&65535
                for prior_id in bands[band].get(key,[]):candidates[prior_id]=retained_simhash[prior_id]
            if any(hamming(fingerprint,prior)<=3 for prior in candidates.values()):raise ValueError("retained near duplicate")
            retained_exact[exact]=row["document_id"]
            for value in row["paragraph_sha256"]:retained_paragraph[value]=row["document_id"]
            retained_simhash[row["document_id"]]=fingerprint
            for band in range(4):bands[band].setdefault((fingerprint>>(16*band))&65535,[]).append(row["document_id"])
    if set(retained_exact.values()) != train_ids|val_ids:raise ValueError("audit does not cover every retained document")
    return {"status":"PASS","raw_documents":len(cache),"audit_decisions":sum(counts.values()),"decision_counts":dict(counts),"recomputed_fingerprints":len(retained_exact)}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("manifest");parser.add_argument("--output",required=True);args=parser.parse_args()
    path=Path(args.manifest).resolve();base=path.parent;manifest=verify_top_manifest(path,TOKENIZER)
    if manifest["corpus_identity"]!=CORPUS_V2R1_ID:raise ValueError("not DATA-D-BROAD-v2r1")
    commit=manifest["builder_git_commit"]
    subprocess.run(["git","cat-file","-e",f"{commit}:scripts/prepare_phase7f_v2.py"],cwd=ROOT,check=True)
    _,train_ids=arrays(base,manifest["shards"],"train");_,val_ids=arrays(base,manifest["validation_shards"],"validation")
    if train_ids&val_ids:raise ValueError("training/validation document overlap")
    if not manifest["common_validation_registry"]["registry_documents"] or manifest["common_validation_registry"]["tokens"]!=500_000:
        raise ValueError("common-validation registry incomplete")
    audit=verify_audit(base,manifest,train_ids,val_ids);resume=exact_resume_probe(base,manifest)
    source_tokens={source:sum(json.loads((base/x["manifest_path"]).read_text())["token_count"] for x in manifest["shards"] if json.loads((base/x["manifest_path"]).read_text())["source"]==source) for source in SOURCES}
    total=sum(source_tokens.values());mixture={source:source_tokens[source]/total for source in SOURCES}
    if any(abs(mixture[source]-{"fineweb_edu":.5,"wikipedia":.25,"fineweb":.25}[source])>.001 for source in SOURCES):raise ValueError("realized mixture outside tolerance")
    report={"corpus_identity":CORPUS_V2R1_ID,"manifest_sha256":sha256_file(path),"builder_git_commit":commit,
        "total_train_tokens":total,"total_train_documents":len(train_ids),"total_validation_documents":len(val_ids),"source_tokens":source_tokens,
        "realized_mixture":mixture,"train_validation_disjointness":"PASS","common_validation_registry":"PASS","shard_and_tokenizer_verification":"PASS",
        "independent_dedup_verification":audit,"full_exact_resume_test":resume,"acceptance_gates":"PASS"}
    Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":main()
