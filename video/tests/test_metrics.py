import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.inject_results import normalize

def test_complete_synthetic_final():
    selection={'candidate_id':'joint-recovery-150000','checkpoint_sha256':'abc123','lineage':'joint','recovery_tokens':150000}
    evaluations={'joint-recovery-150000':{'data_d':6.9,'reasoning_accuracy':.9,'reasoning_margin':5.2,'wikitext_bpb':1.4,'wikitext_ppl':24.2,'arc_easy':.4,'piqa':.59,'hellaswag':.29,'winogrande':.52}}
    data,sid=normalize(selection,evaluations)
    assert sid=='joint-recovery-150000' and data['data_d']==6.9 and data['reasoning_accuracy']==.9

def test_proxy_cannot_masquerade_as_final():
    try:normalize({'candidate_id':'x','checkpoint_sha256':'abc'},{'x':{'proxy_256':{'data_d_validation':4,'v2_ranking_accuracy':.8}}})
    except ValueError:pass
    else:raise AssertionError('proxy accepted as official final')

def test_salvage_schema_and_hash_guard():
    import json
    root=Path(__file__).resolve().parents[1]
    s=json.loads((root/'tests/fixtures/mock_selection.json').read_text())
    e=json.loads((root/'tests/fixtures/mock_evaluations.json').read_text())
    data,sid=normalize(s,e)
    assert data['reasoning_scope']=='proxy_1024'
    assert data['arc_easy']==.41 and data['data_d']==6.91
    e[sid]['checkpoint_sha256']='wrong'
    try:normalize(s,e)
    except ValueError:pass
    else:raise AssertionError('mismatched checkpoint hashes accepted')
