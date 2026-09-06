from collections import Counter
import hashlib
import numpy as np
from latticelm.data_d import SourceStream
from latticelm.lattice_reason.core import FAMILIES
from latticelm.lattice_reason.sampling import TokenBalancedReasonStream,TournamentMixture,lr_slots
from latticelm.tokenizer import load_tokenizer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def data_streams():
 return {s:SourceStream([np.arange(8193,dtype=np.int32)%4096],128,10+i) for i,s in enumerate(("fineweb_edu","wikipedia","fineweb"))}

def test_frozen_source_schedules_are_exact():
 assert {p:sum(lr_slots(p,i) for i in range(5)) for p in (5,10,15,25,50)}=={5:2,10:4,15:6,25:10,50:20}

def test_reason_token_cycle_and_resume():
 tok=load_tokenizer(ROOT/"artifacts/tokenizers/babylm_2026_4k.json");stream=TokenBalancedReasonStream(tok)
 counts=Counter()
 for _ in range(20):_,_,f=stream.one();counts[f]+=128
 assert [counts[f] for f in FAMILIES]==[384,384,384,384,384,256,256,128]
 state=stream.state_dict();expected=stream.one()
 resumed=TokenBalancedReasonStream(tok);resumed.load_state_dict(state);got=resumed.one()
 assert expected[2]==got[2] and np.array_equal(expected[0],got[0])

def test_full_mixture_next_batch_identity():
 tok=load_tokenizer(ROOT/"artifacts/tokenizers/babylm_2026_4k.json");mix=TournamentMixture(data_streams(),TokenBalancedReasonStream(tok),10)
 for _ in range(7):mix.batch()
 state=mix.state_dict();expected=mix.batch()[0]
 resumed=TournamentMixture(data_streams(),TokenBalancedReasonStream(tok),10);resumed.load_state_dict(state)
 assert hashlib.sha256(expected.tobytes()).digest()==hashlib.sha256(resumed.batch()[0].tobytes()).digest()
