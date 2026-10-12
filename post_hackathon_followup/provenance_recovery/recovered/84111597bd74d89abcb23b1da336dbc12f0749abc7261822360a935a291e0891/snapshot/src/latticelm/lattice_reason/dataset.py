"""Immutable int32 shard writer, nested views, guards, and verification."""
from __future__ import annotations

from collections import Counter
import hashlib,json,os,resource,shutil,time
from pathlib import Path
from typing import Any
import numpy as np
from tokenizers import Tokenizer

from .core import (FAMILIES, GENERATOR_VERSION, TRAIN_TEMPLATES, VALID_TEMPLATES,
                   canonical_json, generate_example, verify_example)

MILESTONES=(1_000_000,5_000_000,10_000_000,25_000_000,50_000_000,100_000_000,250_000_000,500_000_000,1_000_000_000)
SCHEMA="lattice-reason-canonical-v1"; SHARD_SCHEMA="lattice-reason-int32-v1"

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def select_adaptive_target(t100_seconds:float,planning_seconds:float=7.5*3600,safety:float=1.15)->int:
    if t100_seconds<=0:raise ValueError("T100 must be positive")
    fits=[m for m in MILESTONES if m>=100_000_000 and t100_seconds*(m/100_000_000)*safety<=planning_seconds]
    return max(fits,default=100_000_000)

def deadline_allows(start:float,now:float,hard_seconds:float,predicted_block_seconds:float,finalize_reserve:float=120)->bool:
    return now-start+predicted_block_seconds+finalize_reserve < hard_seconds

def disk_allows(path:Path,needed_bytes:int,reserve_bytes:int=5<<30)->bool:
    return shutil.disk_usage(path).free-needed_bytes>=reserve_bytes

def _atomic(path:Path,data:bytes)->None:
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_bytes(data);os.replace(tmp,path)

def _view(out:Path,target:int,shards:list[dict],examples:int,families:Counter,difficulties:Counter,tokenizer_hash:str,commit:str)->dict:
    included=[];remaining=target
    for shard in shards:
        if remaining<=0:break
        take=min(remaining,shard["token_count"]);included.append({"path":shard["path"],"sha256":shard["sha256"],"tokens":take});remaining-=take
    if remaining:raise ValueError("insufficient canonical tokens")
    payload={"schema_version":SCHEMA,"generator_version":GENERATOR_VERSION,"generator_commit":commit,"tokenizer_sha256":tokenizer_hash,
             "exact_tokens":target,"canonical_prefix":True,"examples_generated_through_prefix":examples,"shards":included,
             "family_counts_at_freeze":dict(sorted(families.items())),"difficulty_counts_at_freeze":dict(sorted(difficulties.items())),"verification":"PASS"}
    path=out/f"view-{target}.manifest.json";_atomic(path,canonical_json(payload));return {"tokens":target,"path":path.name,"sha256":sha256_file(path)}

def verify_manifest(path:Path,tokenizer_path:Path|None=None)->dict[str,Any]:
    data=json.loads(path.read_text());base=path.parent
    if data.get("schema_version")!=SCHEMA:raise ValueError("unsupported manifest")
    if tokenizer_path and data["tokenizer_sha256"]!=sha256_file(tokenizer_path):raise ValueError("tokenizer identity mismatch")
    total=0
    for item in data["shards"]:
        p=base/item["path"]
        if not p.exists() or sha256_file(p)!=item["sha256"]:raise ValueError("shard hash mismatch")
        if p.stat().st_size//4 < item.get("tokens",item.get("token_count",0)):raise ValueError("short shard")
        total+=item.get("tokens",item.get("token_count",0))
    expected=data.get("exact_tokens",data.get("token_count"))
    if expected is not None and total!=expected:raise ValueError("token count mismatch")
    return data

def _normalized_words(text:str)->list[str]:
    import re
    return re.findall(r"[a-z0-9]+",text.lower())

def compile_contamination_registry(registry:dict[str,list[str]],ngram:int=13)->dict[str,Any]:
    grams:dict[bytes,set[str]]={};exact:dict[str,set[str]]={}
    for name in sorted(registry):
        exact[name]=set()
        for row in registry[name]:
            words=_normalized_words(row)
            if len(words)>=8:exact[name].add(" ".join(words))
            for i in range(max(0,len(words)-ngram+1)):
                key=hashlib.blake2b(" ".join(words[i:i+ngram]).encode(),digest_size=8).digest();grams.setdefault(key,set()).add(name)
    return {"ngram":ngram,"grams":grams,"exact":exact}

def contamination_matches(text:str,registry:dict[str,Any],ngram:int=13)->list[str]:
    """Conservative exact/13-gram guard over a frozen local text registry."""
    words=_normalized_words(text);hits=[]
    if "grams" in registry and "exact" in registry:
        names=set();normalized=" ".join(words)
        for i in range(max(0,len(words)-int(registry["ngram"])+1)):
            key=hashlib.blake2b(" ".join(words[i:i+int(registry['ngram'])]).encode(),digest_size=8).digest();names.update(registry["grams"].get(key,()))
        for name in sorted(names):
            if any(row in normalized for row in registry["exact"].get(name,())) or name in names:hits.append(name)
        return hits
    grams={tuple(words[i:i+ngram]) for i in range(max(0,len(words)-ngram+1))}
    normalized=" ".join(words)
    for name in sorted(registry):
        for row in registry[name]:
            rw=_normalized_words(row);rtext=" ".join(rw)
            if (len(rw)>=8 and rtext in normalized) or (len(rw)>=ngram and grams.intersection(tuple(rw[i:i+ngram]) for i in range(len(rw)-ngram+1))):
                hits.append(name);break
    return hits

def build_validation(out:Path,tokenizer_path:Path,per_family_difficulty:int=40,seed:int=99173,
                     registry:dict[str,list[str]]|None=None)->dict[str,Any]:
    out.mkdir(parents=True,exist_ok=True);tokenizer=Tokenizer.from_file(str(tokenizer_path));records=[];counts=Counter();rejected=Counter();counter=10_000_000
    for family in FAMILIES:
        for difficulty in range(1,7):
            accepted=0
            while accepted<per_family_difficulty:
                ex=generate_example(counter,seed,"validation",family,difficulty,include_trace=(accepted%2==0));counter+=1
                hits=contamination_matches(ex.rendered_text,registry or {})
                if hits:rejected.update(hits);continue
                if not verify_example(ex):rejected["verifier"]+=1;continue
                row=ex.to_dict();row["token_count"]=len(tokenizer.encode(ex.rendered_text).ids);records.append(row);counts[f"{family}:{difficulty}"]+=1;accepted+=1
    if set().union(*TRAIN_TEMPLATES.values()).intersection(set().union(*VALID_TEMPLATES.values())):raise RuntimeError("template leakage")
    data=b"".join(canonical_json(x) for x in records);path=out/"validation.jsonl";_atomic(path,data)
    manifest={"schema_version":"lattice-reason-validation-v1","generator_version":GENERATOR_VERSION,"seed":seed,"examples":len(records),"counts":dict(counts),"held_out_templates":True,"held_out_vocabulary":True,"rejected":dict(rejected),"data_path":path.name,"data_sha256":sha256_file(path),"tokenizer_sha256":sha256_file(tokenizer_path),"verification":"PASS"}
    _atomic(out/"validation.manifest.json",canonical_json(manifest));return manifest

def build_canonical(out:Path,tokenizer_path:Path,target:int,seed:int=7319,shard_tokens:int=1_000_000,
                    generator_commit:str="UNCOMMITTED",soft_seconds:float=7.5*3600,hard_seconds:float=8*3600,
                    adaptive:bool=False,min_free_bytes:int=5<<30,fixture_metadata:bool=False,
                    contamination_registry:dict[str,Any]|None=None)->dict[str,Any]:
    if target not in MILESTONES:raise ValueError("target must be a canonical milestone")
    out.mkdir(parents=True,exist_ok=True); tokenizer=Tokenizer.from_file(str(tokenizer_path));tok_hash=sha256_file(tokenizer_path)
    state_path=out/"generation-state.json";started=time.monotonic();wall_started=time.time();shards=[];index=0;tokens=0;families=Counter();diffs=Counter();discarded=Counter();views=[]
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state["generator_version"]!=GENERATOR_VERSION or state["seed"]!=seed or state["tokenizer_sha256"]!=tok_hash:raise ValueError("resume identity mismatch")
        shards=state["shards"];index=state["next_example_index"];tokens=state["token_count"];families.update(state["family_counts"]);diffs.update({int(k):v for k,v in state["difficulty_counts"].items()})
        views=state.get("views",[])
        for s in shards:
            if sha256_file(out/s["path"])!=s["sha256"]:raise ValueError("resume shard corruption")
    metadata_handle=(out/"examples.fixture.jsonl").open("a") if fixture_metadata else None
    first100_seconds=None;actual_target=target;last_rate=None
    try:
        while tokens<actual_target:
            block_goal=min(shard_tokens,actual_target-tokens);estimated=(block_goal/(last_rate or 25_000))
            if not deadline_allows(started,time.monotonic(),hard_seconds,estimated):break
            if time.monotonic()-started>=soft_seconds:break
            if not disk_allows(out,block_goal*4+64_000_000,min_free_bytes):break
            buf=[];block_families=Counter();block_diffs=Counter();first=index
            while len(buf)<block_goal:
                ex=generate_example(index,seed,"train",include_trace=(index%4==0));index+=1
                if not verify_example(ex):discarded["verifier"]+=1;continue
                hits=contamination_matches(ex.rendered_text,contamination_registry or {})
                if hits:discarded.update("contamination:"+x for x in hits);continue
                ids=tokenizer.encode(ex.rendered_text).ids
                if not ids:discarded["empty"]+=1;continue
                buf.extend(ids);block_families[ex.family]+=1;block_diffs[ex.difficulty]+=1
                if metadata_handle:metadata_handle.write(json.dumps(ex.to_dict(),sort_keys=True)+"\n")
            buf=buf[:block_goal];shard_index=len(shards);partial=out/f"shard-{shard_index:06d}.int32.partial";final=out/f"shard-{shard_index:06d}.int32"
            np.asarray(buf,dtype="<i4").tofile(partial);os.replace(partial,final);digest=sha256_file(final)
            elapsed=max(time.monotonic()-started,1e-9);last_rate=(tokens+len(buf))/elapsed
            entry={"schema_version":SHARD_SCHEMA,"path":final.name,"sha256":digest,"token_count":len(buf),"example_id_range":[f"lr-t-{first:012d}",f"lr-t-{index-1:012d}"],"example_count":index-first,"family_counts":dict(block_families),"difficulty_counts":dict(block_diffs),"generator_commit":generator_commit,"tokenizer_sha256":tok_hash}
            shards.append(entry);tokens+=len(buf);families.update(block_families);diffs.update(block_diffs)
            for milestone in MILESTONES:
                if milestone<=tokens and not any(v["tokens"]==milestone for v in views):views.append(_view(out,milestone,shards,index,families,diffs,tok_hash,generator_commit))
            state={"schema_version":SCHEMA,"generator_version":GENERATOR_VERSION,"generator_commit":generator_commit,"seed":seed,"tokenizer_sha256":tok_hash,"token_count":tokens,"next_example_index":index,"shards":shards,"views":views,"family_counts":families,"difficulty_counts":diffs,"discarded":discarded}
            _atomic(state_path,canonical_json(state))
            if tokens==100_000_000 and first100_seconds is None:
                first100_seconds=time.monotonic()-started
                if adaptive:actual_target=max(actual_target,select_adaptive_target(first100_seconds,soft_seconds))
    finally:
        if metadata_handle:metadata_handle.close()
        for partial in out.glob("*.partial"):partial.unlink()
    elapsed=time.monotonic()-started;top={"schema_version":SCHEMA,"generator_version":GENERATOR_VERSION,"generator_commit":generator_commit,"seed":seed,"tokenizer_sha256":tok_hash,"token_count":tokens,"example_count":index,"shards":shards,"views":views,"family_counts":dict(sorted(families.items())),"difficulty_counts":dict(sorted(diffs.items())),"discarded":dict(discarded),"wall_started":wall_started,"wall_seconds":elapsed,"tokens_per_second":tokens/max(elapsed,1e-9),"examples_per_second":index/max(elapsed,1e-9),"first_100m_wall_seconds":first100_seconds,"selected_target":actual_target,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"disk_bytes":sum((out/s["path"]).stat().st_size for s in shards),"free_disk_bytes":shutil.disk_usage(out).free,"status":"COMPLETE" if tokens==actual_target else "SAFE_BOUNDARY_STOP"}
    _atomic(out/"manifest.json",canonical_json(top));return top
