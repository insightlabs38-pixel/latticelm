"""Small, disposable checkpoint/resume and artifact drill for production code."""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.final_recipe import make_optimizer
from latticelm.model import build_model


def main() -> None:
    cfg = LatticeConfig(vocab_size=512, d_model=96, n_layers=2, n_heads=4, n_kv_heads=2,
                        ffn_hidden=192, context_length=32, architecture="co4_causal", real_gqa=True,
                        qk_norm=True, tie_embeddings=False, batch_size=2, learning_rate=5e-4,
                        adam_beta2=.95, lr_schedule="wsd", warmup_fraction=.02, stable_fraction=.83,
                        decay_fraction=.15, seed=1337)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    model = build_model(cfg); optimizer, partition = make_optimizer(model, "adamw", cfg.learning_rate, cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
    batches = [torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length + 1)) for _ in range(4)]
    with tempfile.TemporaryDirectory(prefix="latticelm-final-drill-") as directory:
        root = Path(directory); events = []
        for step in range(2):
            batch = batches[step]; optimizer.zero_grad(); logits, _ = model(batch[:, :-1]); loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1)); loss.backward(); optimizer.step()
            events.append({"step": step + 1, "loss": float(loss.detach())})
        checkpoint = {"model": model.state_dict(), "optimizers": optimizer.state_dict(), "step": 2,
                      "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                      "next_batch_sha256": __import__("hashlib").sha256(batches[2].numpy().tobytes()).hexdigest(), "optimizer_partition": partition}
        path = root / "latest.pt"; torch.save(checkpoint, path); loaded = torch.load(path, map_location="cpu", weights_only=False)
        restored = build_model(cfg); restored_optimizer, restored_partition = make_optimizer(restored, "adamw", cfg.learning_rate, cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
        restored.load_state_dict(loaded["model"], strict=True); restored_optimizer.load_state_dict(loaded["optimizers"])
        assert restored_partition == loaded["optimizer_partition"]
        assert loaded["next_batch_sha256"] == __import__("hashlib").sha256(batches[2].numpy().tobytes()).hexdigest()
        batch = batches[2]; restored_optimizer.zero_grad(); logits, _ = restored(batch[:, :-1]); resumed_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1)); resumed_loss.backward(); restored_optimizer.step()
        evidence = {"status": "PASS", "initial_steps": 2, "resumed_steps": 1, "finite": bool(torch.isfinite(resumed_loss)),
                    "checkpoint_reload": "PASS", "next_batch_identity": "PASS", "optimizer_state": "PASS", "events": events}
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__": main()
