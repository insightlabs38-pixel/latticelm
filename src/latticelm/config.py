from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class LatticeConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: int = 1
    ffn_hidden: int = 768
    context_length: int = 128
    dropout: float = 0.0
    memory_enabled: bool = False
    memory_slots: int = 65536
    memory_dim: int = 64
    architecture: str = "lattice"
    mixer_strategy: str = "attention"
    local_attention_window: int = 32
    batch_size: int = 8
    max_steps: int = 600
    eval_interval: int = 100
    checkpoint_interval: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_schedule: str = "constant"
    seed: int = 1337
    num_threads: int = 2
    memory_heads: int = 2
    memory_orders: tuple[int, ...] = (2, 3)
    memory_conv_kernel: int = 4
    memory_conv_enabled: bool = True
    memory_insert_layers: tuple[int, ...] = (1,)
    memory_token_map_path: str | None = None
    memory_dropout: float = 0.0
    memory_lr_multiplier: float = 0.3
    tie_embeddings: bool = True
    hf_persistence_enabled: bool = False
    hf_upload_interval_tokens: int = 1_048_576
    hf_best_upload_interval_tokens: int = 1_048_576
    hf_update_best: bool = False
    hf_named_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model must divide n_heads and n_heads must divide n_kv_heads")
        if self.context_length < 2 or self.memory_slots < 1:
            raise ValueError("invalid context or memory size")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        if self.mixer_strategy not in {"attention", "hybrid"}:
            raise ValueError("mixer_strategy must be attention or hybrid")
        if self.architecture not in {"lattice", "mini_engram", "co4_causal", "co4_memory", "co4_inspired", "sllama_inspired"}:
            raise ValueError("unsupported architecture")
        self.memory_orders = tuple(self.memory_orders)
        self.memory_insert_layers = tuple(self.memory_insert_layers)

    @classmethod
    def from_json(cls, path: str | Path) -> "LatticeConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)
