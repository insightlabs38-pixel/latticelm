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

def build(output:Path,total_tokens:int,hard_deadline_epoch=None):
    if total_tokens<1_500_000_000:raise ValueError("v4 requires at least 1.5B newly certified tokens")
    # Reuse the audited implementation, but seed exact/paragraph/SimHash indexes
    # from v3 and stamp a distinct corpus identity throughout child manifests.
    builder.OLD=OLD;builder.CORPUS_V3_ID=IDENTITY
    top=builder.build(output,total_tokens,hard_deadline_epoch)
    top["schema_version"]="data-d-corpus-v1";top["corpus_identity"]=IDENTITY
    top["predecessor_manifest_sha256"]=builder.sha256_file(OLD/"manifest.json")
    top["deduplicated_against"]=["DATA-C","DATA-D-BROAD-v2r1","DATA-D-BROAD-v3","validation registries","competition evaluation fingerprints"]
    top["dataset_transition_policy"]="new deterministic stream at the 900M boundary"
    (output/"manifest.json").write_bytes(builder.canonical_json(top))
    return top
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--total-tokens",type=int,default=2_250_000_000);p.add_argument("--hard-deadline-epoch",type=float);a=p.parse_args();print(json.dumps(build(a.output,a.total_tokens,a.hard_deadline_epoch)))
if __name__=="__main__":main()
