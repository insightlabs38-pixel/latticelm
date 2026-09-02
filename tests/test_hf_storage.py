import json

import torch

from latticelm.config import LatticeConfig
from latticelm.hf_storage import export_checkpoint, sha256
from latticelm.model import build_model


def test_safetensors_export_roundtrip_and_manifest(tmp_path) -> None:
    config = LatticeConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                           ffn_hidden=64, context_length=16, memory_enabled=False)
    torch.manual_seed(17)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 3,
                "config": config.to_dict(), "torch_rng_state": torch.get_rng_state(),
                "data_generator_state": torch.Generator().manual_seed(1).get_state()}, checkpoint)
    tokenizer = tmp_path / "tokenizer.json"; tokenizer.write_text("{}")
    tokenizer_report = tmp_path / "tokenizer.report.json"; tokenizer_report.write_text("{}")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"dataset": "test", "revision": "abc"}))
    output = tmp_path / "export"
    manifest = export_checkpoint(checkpoint, output, "tiny", "latest", tokenizer,
                                 tokenizer_report, dataset,
                                 {"tokens_trained": 48, "wall_seconds": 1.0,
                                  "val_loss": 2.0, "val_ppl": 7.389})
    assert manifest["safetensors_roundtrip_verified"] is True
    assert manifest["total_trainable_parameters"] == sum(p.numel() for p in model.parameters())
    assert (output / "model.safetensors").is_file()
    assert sha256(output / "model.safetensors") in (output / "SHA256SUMS").read_text()
    resume = torch.load(output / "resume_state.pt", weights_only=False)
    assert "model" not in resume
    assert resume["exact_rng_state_available"] is True
