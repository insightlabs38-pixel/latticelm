"""Build DATA-D-v4 as new material globally deduplicated against DATA-D-v3.

The v4 stream deliberately contains newly accepted documents only.  This makes the
900M boundary semantics unambiguous: the optimizer lineage continues, while a new
deterministic selector starts over a corpus whose dedup index was seeded by v3.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
try:import build_data_d_v3 as builder
except ModuleNotFoundError:import scripts.build_data_d_v3 as builder

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"artifacts/data/data_d_v3/canonical-1b"
IDENTITY="DATA-D-BROAD-v4"
TOKENIZER=ROOT/"artifacts/tokenizers/final_corpus_4k.json"
FINEMATH_REVISION="e92b25a616738fe95dc186b64dfb19f9c8525594"
MIXTURE={"fineweb_edu":.45,"wikipedia":.25,"fineweb":.20,"finemath":.10}

def build(output:Path,total_tokens:int,hard_deadline_epoch=None,tokenizer:Path=TOKENIZER):
    if total_tokens<1_500_000_000:raise ValueError("v4 requires at least 1.5B newly certified tokens")
    # Reuse the audited implementation, but seed exact/paragraph/SimHash indexes
    # from v3 and stamp a distinct corpus identity throughout child manifests.
    builder.OLD=OLD;builder.CORPUS_V3_ID=IDENTITY;builder.TOKENIZER=tokenizer
    builder.SOURCES=tuple(MIXTURE);builder.MIXTURE=MIXTURE
    builder.REVISIONS={**builder.REVISIONS,"finemath":("HuggingFaceTB/finemath","finemath-4plus",FINEMATH_REVISION)}
    builder.LICENSES={**builder.LICENSES,"finemath":"ODC-By-1.0"}
    # The 96 GiB research VM already retains immutable v3 and WSD evidence.
    # Four GiB is the measured safe floor; the master separately budgets its
    # rotating research checkpoints and never creates a second v4 copy.
    top=builder.build(output,total_tokens,hard_deadline_epoch,reserve=4*1024**3)
    top["schema_version"]="data-d-corpus-v1";top["corpus_identity"]=IDENTITY
    top["predecessor_manifest_sha256"]=builder.sha256_file(OLD/"manifest.json")
    top["deduplicated_against"]=["DATA-C","DATA-D-BROAD-v2r1","DATA-D-BROAD-v3","validation registries","competition evaluation fingerprints"]
    top["dataset_transition_policy"]="fresh final pretraining corpus, globally deduplicated against predecessor corpora"
    top["source_policy"]={"mixture":MIXTURE,"finemath_configuration":"finemath-4plus","synthetic_model_data":"excluded","evaluation_examples":"quarantined by registry"}
    (output/"manifest.json").write_bytes(builder.canonical_json(top))
    return top
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--total-tokens",type=int,default=2_250_000_000);p.add_argument("--hard-deadline-epoch",type=float);p.add_argument("--tokenizer",type=Path,default=TOKENIZER);a=p.parse_args();print(json.dumps(build(a.output,a.total_tokens,a.hard_deadline_epoch,a.tokenizer)))
if __name__=="__main__":main()
