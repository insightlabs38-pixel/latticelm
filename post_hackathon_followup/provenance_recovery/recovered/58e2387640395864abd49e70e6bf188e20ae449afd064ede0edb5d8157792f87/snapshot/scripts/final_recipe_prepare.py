"""Deterministic tokenizer research and official-task context audit."""
from __future__ import annotations

import argparse, hashlib, json, math, os, statistics, time
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from latticelm.tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/data/phase7f_v2r1/canonical-100m/normalized-source-cache.jsonl"
TOKDIR = ROOT / "artifacts/tokenizers"
TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")
FINEMATH_REVISION="e92b25a616738fe95dc186b64dfb19f9c8525594"
MATH_SAMPLE=ROOT/"artifacts/tokenizer_sources/finemath-4plus-40mb.jsonl"


def atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def rows():
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line); yield item["source"], item["normalized_text"]


def ensure_math_sample() -> None:
    if MATH_SAMPLE.exists(): return
    from datasets import load_dataset
    MATH_SAMPLE.parent.mkdir(parents=True,exist_ok=True);partial=MATH_SAMPLE.with_suffix(".partial")
    stream=load_dataset("HuggingFaceTB/finemath","finemath-4plus",split="train",streaming=True,revision=FINEMATH_REVISION)
    total=0
    with partial.open("w",encoding="utf-8") as handle:
        for item in stream:
            text=item.get("text","");record=json.dumps({"source":"finemath","normalized_text":text},sort_keys=True)+"\n";handle.write(record);total+=len(record.encode())
            if total>=40_000_000:break
        handle.flush();os.fsync(handle.fileno())
    os.replace(partial,MATH_SAMPLE)


def final_corpus_rows():
    yield from rows()
    with MATH_SAMPLE.open(encoding="utf-8") as handle:
        for line in handle:
            item=json.loads(line);yield item["source"],item["normalized_text"]


def train_tokenizers() -> dict:
    ensure_math_sample()
    result = {}
    samples: dict[str, list[str]] = defaultdict(list)
    for source, text in final_corpus_rows():
        if sum(map(len, samples[source])) < 2_000_000: samples[source].append(text)
    math_samples = ["If x = 12.50 and y = 3, then x/y = 4.1667. 1234567890.\n",
                    "Proof: for every integer n >= 2, n^2 - n is even.\n"]
    code_samples = ["def f(x: int) -> int:\n    return x * x + 1\n", "SELECT id FROM items WHERE score >= 10;\n"]
    for vocab in (4096, 8192):
        started = time.perf_counter(); tokenizer = Tokenizer(BPE(unk_token="<unk>")); tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]); tokenizer.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(vocab_size=vocab, min_frequency=2,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tokenizer.train_from_iterator((text for _, text in final_corpus_rows()), trainer=trainer)
        if tokenizer.get_vocab_size() != vocab: raise RuntimeError("tokenizer did not reach requested vocabulary")
        path = TOKDIR / f"final_corpus_{vocab//1024}k.json"; path.parent.mkdir(parents=True, exist_ok=True); tokenizer.save(str(path))
        metrics = {}
        domains = {**{k:"\n".join(v) for k,v in samples.items()}, "math": "".join(math_samples), "code_like":"".join(code_samples)}
        for domain, text in domains.items():
            ids = tokenizer.encode(text).ids; words = text.split(); raw = text.encode()
            metrics[domain] = {"bytes_per_token":len(raw)/max(1,len(ids)), "chars_per_token":len(text)/max(1,len(ids)),
                "tokens_per_word":len(ids)/max(1,len(words)), "characters":len(text), "tokens":len(ids)}
        numerics = {s: tokenizer.encode(s).tokens for s in ("1234567890", "3.14159265", "-42", "1e-9", "2026-10-01")}
        report={"schema":"final-tokenizer-report-v1","vocab_size":vocab,"sha256":sha256(path),"source_cache":str(CACHE),
            "source_cache_sha256":sha256(CACHE),"finemath_sample":str(MATH_SAMPLE),"finemath_sample_sha256":sha256(MATH_SAMPLE),"finemath_revision":FINEMATH_REVISION,"pretokenizer":"Digits(individual_digits=True) then ByteLevel","normalizer":"NFKC",
            "metrics":metrics,"numeric_tokenization":numerics,"training_seconds":time.perf_counter()-started,
            "parameter_cost":{"untied_at_d552":2*vocab*552,"tied_at_d552":vocab*552}}
        atomic(path.with_suffix(".report.json"),report); result[str(vocab)]=report
    # Offline rule: 8K must improve weighted bytes/token by >=12% after charging
    # its extra 4.5M untied parameters; otherwise retain 4K.
    def weighted(v):
        m=result[str(v)]["metrics"]; return sum(m[s]["bytes_per_token"]*w for s,w in (("fineweb_edu",.45),("wikipedia",.25),("fineweb",.2),("math",.1)))
    b4,b8=weighted(4096),weighted(8192); gain=b8/b4-1
    selected=8192 if gain>=.12 else 4096
    decision={"schema":"tokenizer-decision-v1","selected_vocab_size":selected,"selected":str(TOKDIR/f"final_corpus_{selected//1024}k.json"),
        "weighted_bytes_per_token":{"4096":b4,"8192":b8},"relative_compression_gain_8k":gain,
        "rule":"select 8K only for >=12% weighted compression gain because untied 8K costs ~4.5M extra parameters at d=552",
        "training_ablation_authorized":False,"reports":result}
    atomic(ROOT/"artifacts/tokenizer_decision.json",decision); return decision


def percentile(values: list[int], q: float) -> int:
    if not values:return 0
    return sorted(values)[math.ceil(q*len(values))-1]


def context_audit() -> dict:
    from lm_eval.tasks import TaskManager, get_task_dict
    tokenizers={p.stem:load_tokenizer(p) for p in (TOKDIR/"final_corpus_4k.json",TOKDIR/"final_corpus_8k.json")}
    tasks=get_task_dict(list(TASKS),task_manager=TaskManager()); out={}
    for task_name, task in tasks.items():
        task.build_all_requests(rank=0,world_size=1,cache_requests=True,rewrite_requests_cache=False,tokenizer_name="latticelm-final-audit")
        texts=[]
        for instance in task.instances:
            if instance.request_type != "loglikelihood": continue
            context, continuation=instance.args; texts.append(context+continuation)
        out[task_name]={"task_version":task.VERSION,"requests":len(texts),"tokenizers":{}}
        for name,tok in tokenizers.items():
            lengths=[len(tok.encode(x)) for x in texts]; limits={}
            for limit in (128,256,512,1024):
                lost=[max(0,x-limit) for x in lengths]
                limits[str(limit)]={"fraction_truncated":sum(x>0 for x in lost)/max(1,len(lost)),"average_tokens_lost":sum(lost)/max(1,len(lost))}
            out[task_name]["tokenizers"][name]={"median":statistics.median(lengths),"p90":percentile(lengths,.9),"p95":percentile(lengths,.95),
                "p99":percentile(lengths,.99),"maximum":max(lengths,default=0),"limits":limits}
    selected=json.loads((ROOT/"artifacts/tokenizer_decision.json").read_text()); key=f"final_corpus_{selected['selected_vocab_size']//1024}k"
    aggregate=[]
    for task in out.values():
        # Reconstruct conservative aggregate decision from request-weighted task summaries.
        aggregate.append(task["tokenizers"][key]["limits"])
    frac256=max(x["256"]["fraction_truncated"] for x in aggregate); frac512=max(x["512"]["fraction_truncated"] for x in aggregate)
    context=256 if frac256<=.01 else 512
    if frac512>.01: context=1024
    decision={"schema":"context-decision-v1","selected_context":context,"smallest_practical_rule":"max per-task truncation fraction <=1%",
        "max_task_fraction_truncated":{"256":frac256,"512":frac512},"attention_cost_relative_to_128":{"256":2,"512":4,"1024":8},"tasks":out}
    atomic(ROOT/"artifacts/context_decision.json",decision); return decision


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("tokenizers","context"));a=p.parse_args()
    print(json.dumps(train_tokenizers() if a.mode=="tokenizers" else context_audit(),indent=2))
if __name__=="__main__":main()
