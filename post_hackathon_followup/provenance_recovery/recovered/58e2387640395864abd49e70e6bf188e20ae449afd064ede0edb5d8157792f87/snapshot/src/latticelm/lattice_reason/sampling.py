"""Deterministic loss-token-balanced LatticeReason and source mixing streams."""
from __future__ import annotations

from collections import Counter
from typing import Mapping
import numpy as np

from .core import FAMILIES, MIXTURE_CYCLE, generate_example, verify_example


class FamilyTokenStream:
    """Generate a continuous token stream for one family with exact resume state."""
    def __init__(self, family: str, tokenizer, context: int = 128, seed: int = 7319):
        if family not in FAMILIES: raise ValueError(f"unknown family: {family}")
        self.family, self.tokenizer, self.context, self.seed = family, tokenizer, context, seed
        self.next_index = 0; self.buffer: list[int] = []; self.draws = 0

    def _fill(self, needed: int) -> None:
        while len(self.buffer) < needed:
            example = generate_example(self.next_index, self.seed, "train", self.family,
                                       1 + self.next_index % 6, include_trace=self.next_index % 4 == 0)
            self.next_index += 1
            if not verify_example(example): raise RuntimeError("LatticeReason verifier failure")
            self.buffer.extend(self.tokenizer.encode(example.rendered_text))

    def one(self) -> tuple[np.ndarray, np.ndarray]:
        self._fill(self.context + 1); values = self.buffer[:self.context + 1]
        del self.buffer[:self.context]; self.draws += 1
        array = np.asarray(values, dtype=np.int64)
        return array[:-1], array[1:]

    def state_dict(self) -> dict:
        return {"next_index": self.next_index, "buffer": list(self.buffer), "draws": self.draws}

    def load_state_dict(self, state: Mapping) -> None:
        self.next_index=int(state["next_index"]); self.buffer=list(state["buffer"]); self.draws=int(state["draws"])


class TokenBalancedReasonStream:
    """20-sequence cycle gives exact 15/15/15/15/15/10/10/5 token shares."""
    def __init__(self, tokenizer, context: int = 128, seed: int = 7319):
        self.streams={f:FamilyTokenStream(f,tokenizer,context,seed+i*100_003) for i,f in enumerate(FAMILIES)}
        self.counter=0

    def one(self) -> tuple[np.ndarray,np.ndarray,str]:
        family=MIXTURE_CYCLE[self.counter % len(MIXTURE_CYCLE)]; self.counter += 1
        x,y=self.streams[family].one(); return x,y,family

    def state_dict(self) -> dict:
        return {"family_selector_counter":self.counter,"family_streams":{f:s.state_dict() for f,s in self.streams.items()}}

    def load_state_dict(self,state:Mapping)->None:
        self.counter=int(state["family_selector_counter"])
        for f,s in self.streams.items():s.load_state_dict(state["family_streams"][f])


def lr_slots(percent: int, batch_index: int) -> int:
    """Frozen five-batch schedules; every block has 40 sequences."""
    schedules={5:(1,0,1,0,0),10:(1,1,1,1,0),15:(2,1,1,1,1),25:(2,2,2,2,2),50:(4,4,4,4,4)}
    if percent not in schedules: raise ValueError("unsupported frozen mixture")
    return schedules[percent][batch_index % 5]


class TournamentMixture:
    """Mix DATA-D and token-balanced reasoning without random source selection."""
    DATA_CYCLE=("fineweb_edu","fineweb_edu","wikipedia","fineweb")
    def __init__(self,data_streams:Mapping,reason_stream:TokenBalancedReasonStream,lr_percent:int):
        self.data_streams=dict(data_streams); self.reason=reason_stream; self.lr_percent=lr_percent
        self.batch_counter=0; self.data_counter=0

    def batch(self):
        n_lr=lr_slots(self.lr_percent,self.batch_counter); labels=[]; families=[]; pairs=[]
        for _ in range(8-n_lr):
            label=self.DATA_CYCLE[self.data_counter%4];self.data_counter+=1
            pairs.append(self.data_streams[label].one());labels.append("DATA-D");families.append(None)
        for _ in range(n_lr):
            x,y,family=self.reason.one();pairs.append((x,y));labels.append("LatticeReason");families.append(family)
        self.batch_counter+=1
        return np.stack([x for x,_ in pairs]),np.stack([y for _,y in pairs]),tuple(labels),tuple(families)

    def state_dict(self)->dict:
        return {"batch_counter":self.batch_counter,"data_counter":self.data_counter,"lr_percent":self.lr_percent,
                "data_streams":{k:v.draws for k,v in self.data_streams.items()},"reason":self.reason.state_dict()}

    def load_state_dict(self,state:Mapping)->None:
        if int(state["lr_percent"])!=self.lr_percent:raise ValueError("resume mixture mismatch")
        self.batch_counter=int(state["batch_counter"]);self.data_counter=int(state["data_counter"])
        for k,v in self.data_streams.items():v.draws=int(state["data_streams"][k])
        self.reason.load_state_dict(state["reason"])

    @staticmethod
    def realized(labels,families,increment:int):
        counts=Counter(); family_counts=Counter();remaining=increment
        for source,family in zip(labels,families):
            used=min(128,remaining);remaining-=used;counts[source]+=used
            if family:family_counts[family]+=used
            if not remaining:break
        return counts,family_counts
