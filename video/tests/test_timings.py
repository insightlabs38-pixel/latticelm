import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config.timings import SCENES,TARGET_SECONDS,MAX_SECONDS

def test_timing_budget():
    assert TARGET_SECONDS==298<MAX_SECONDS
    n=json.loads((Path(__file__).resolve().parents[1]/'config/narration.json').read_text())
    assert [x['scene'] for x in n]==[x for x,_ in SCENES]
    for segment,(_,duration) in zip(n,SCENES):assert segment['estimated_duration']<=duration

from components.timing import allocate
def test_actual_audio_reclaims_scene_holds():
    base={'A':8,'B':74,'C':15};minimum={'A':6,'B':62,'C':4}
    out=allocate(base,minimum,{'A':9},limit=97)
    assert out['A']==9.35 and out['C']<15 and sum(out.values())<=97
