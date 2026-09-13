from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

from latticelm.co4_reference import CausalCo4Attention, rms_unit
from latticelm.config import LatticeConfig
from latticelm.model import apply_rope, build_model
from latticelm.final_recipe import make_optimizer, muon_parameter_partition, wsd_lr_scale
from scripts.evaluate_gibc import LatticeHarnessLM
import scripts.train_final_recipe_experiment as research_trainer
from latticelm.data_d import SCHEMA_VERSION, sha256_file


class BoundaryTokenizer:
    bos_id = 1

    def encode(self, text: str) -> list[int]:
        # Deliberately has a merge spanning "a" + "b".
        out, i = [], 0
        while i < len(text):
            if text[i : i + 2] == "ab":
                out.append(9); i += 2
            else:
                out.append(20 + ord(text[i])); i += 1
        return out


def test_evaluator_tokenizes_joined_causal_text_at_bpe_boundary() -> None:
    harness = object.__new__(LatticeHarnessLM)
    harness.tok = BoundaryTokenizer()
    context, continuation = harness._encode_pair("a", "b")
    assert context == [20 + ord("a")]
    assert continuation == []  # canonical lm-eval split of joined [ab] tokenization
    assert harness._encode("a") + harness._encode("b") != harness._encode("ab")


def test_evaluator_moves_context_trailing_space_to_continuation() -> None:
    harness = object.__new__(LatticeHarnessLM)
    harness.tok = BoundaryTokenizer()
    assert harness._encode_pair("a ", "b") == harness._encode_pair("a", " b")


def config(qk_norm: bool = False) -> LatticeConfig:
    return LatticeConfig(vocab_size=64, d_model=48, n_layers=1, n_heads=6,
                         n_kv_heads=2, ffn_hidden=96, context_length=16,
                         architecture="co4_causal", qk_norm=qk_norm, real_gqa=True)


def test_co4_real_gqa_shapes_parameter_savings_and_gradients() -> None:
    module = CausalCo4Attention(config())
    assert module.q_proj.weight.shape == (48, 48)
    assert module.k_proj.weight.shape == (16, 48)
    assert module.v_proj.weight.shape == (16, 48)
    assert module.latent_k.shape == (1, 2, 1, 8)
    x = torch.randn(2, 7, 48, requires_grad=True)
    module(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_co4_gqa_matches_explicit_reference() -> None:
    torch.manual_seed(4)
    module = CausalCo4Attention(config(qk_norm=True)).double()
    x = torch.randn(2, 6, 48, dtype=torch.double, requires_grad=True)
    actual = module(x)
    b, t, _ = x.shape
    q = module.mod(module.latent_q, module.q_proj(x).view(b, t, 6, 8).transpose(1, 2))
    k = module.mod(module.latent_k, module.k_proj(x).view(b, t, 2, 8).transpose(1, 2))
    v = module.mod(module.latent_v, module.v_proj(x).view(b, t, 2, 8).transpose(1, 2))
    q, k = apply_rope(rms_unit(q)), apply_rope(rms_unit(k))
    expected = F.scaled_dot_product_attention(q, k.repeat_interleave(3, 1),
                                               v.repeat_interleave(3, 1), is_causal=True)
    expected = module.output(expected.transpose(1, 2).contiguous().view(b, t, 48))
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-8)
    ga = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), x)[0]
    assert torch.allclose(ga, ge, atol=1e-10, rtol=1e-8)


def test_qknorm_off_and_on_are_deterministic_serializable_and_causal(tmp_path) -> None:
    for enabled in (False, True):
        torch.manual_seed(8)
        model = build_model(config(enabled)).eval()
        tokens = torch.randint(0, 64, (1, 12))
        a, _ = model(tokens)
        b, _ = model(tokens)
        assert torch.equal(a, b) and torch.isfinite(a).all()
        changed = tokens.clone(); changed[:, 8:] = torch.randint(0, 64, (1, 4))
        c, _ = model(changed)
        assert torch.allclose(a[:, :8], c[:, :8], atol=1e-6)
        path = tmp_path / f"{enabled}.pt"
        torch.save(model.state_dict(), path)
        clone = build_model(config(enabled)); clone.load_state_dict(torch.load(path, weights_only=True))
        d, _ = clone.eval()(tokens)
        assert torch.equal(a, d)


def test_co4_compile_forward_backward_smoke() -> None:
    model = build_model(config(True))
    compiled = torch.compile(model, backend="eager")
    x = torch.randint(0, 64, (2, 8)); y = torch.randint(0, 64, (2, 8))
    _, loss = compiled(x, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_wsd_schedule_landmarks_and_resume_boundary() -> None:
    assert wsd_lr_scale(1, 100) == .5
    assert wsd_lr_scale(2, 100) == 1
    assert wsd_lr_scale(50, 100) == 1
    assert wsd_lr_scale(85, 100) == 1
    assert 0 < wsd_lr_scale(92, 100) < 1
    assert wsd_lr_scale(100, 100) == 0
    assert wsd_lr_scale(86, 100) == wsd_lr_scale(86, 100)


def test_muon_partition_exact_once_and_state_round_trip() -> None:
    model = build_model(config())
    muon, adam, names = muon_parameter_partition(model)
    assert muon and adam and len(names) == len(list(model.named_parameters()))
    bundle, _ = make_optimizer(model, "muon_hybrid", 8e-4, .1, (.9, .95))
    x = torch.randint(0, 64, (2, 8)); y = torch.randint(0, 64, (2, 8))
    _, loss = model(x, y); loss.backward(); bundle.step()
    state = bundle.state_dict()
    clone = build_model(config()); other, _ = make_optimizer(clone, "muon_hybrid", 8e-4, .1, (.9, .95))
    other.load_state_dict(state)
    assert len(other.state_dict()) == 2


def test_real_research_trainer_smoke_trajectory_checkpoint_and_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(research_trainer, "ROOT", tmp_path)
    tokenizer = Path("artifacts/tokenizers/babylm_2026_4k.json").resolve()
    data = tmp_path / "data"; data.mkdir(); values=np.arange(4096,dtype="<i4")%64
    children=[]
    for split in ("train","validation"):
        binary=data/f"{split}.int32";values.tofile(binary)
        manifest={"schema_version":SCHEMA_VERSION,"path":binary.name,"sha256":sha256_file(binary),"token_count":len(values),
                  "source":"fineweb_edu","tokenizer_sha256":sha256_file(tokenizer),"stable_document_ids":[split],
                  "document_count":1,"document_boundary_offsets":[0,len(values)]}
        mp=data/f"{split}.manifest.json";mp.write_text(json.dumps(manifest));children.append({"manifest_path":mp.name,"manifest_sha256":sha256_file(mp)})
    top={"schema_version":"data-d-corpus-v1","corpus_identity":"DATA-D-BROAD-v4","tokenizer_sha256":sha256_file(tokenizer),
         "mixture_definition":{"fineweb_edu":1.0},"total_unique_tokens":len(values),"total_documents":1,
         "shards":[children[0]],"validation_shards":[children[1]]}
    manifest=data/"manifest.json";manifest.write_text(json.dumps(top))
    cfg=config();cfg.batch_size=2;cfg.context_length=8;cfg.vocab_size=64;cfg.max_steps=10
    config_path=tmp_path/"config.json";config_path.write_text(json.dumps(cfg.to_dict()))
    args=SimpleNamespace(config=config_path,threads=2,manifest=manifest,tokenizer=tokenizer,backend="eager",experiment="smoke",
        fresh=True,resume=False,target_tokens=256,checkpoint_tokens=128,parent_decision="integration-test")
    result=research_trainer.run(args)
    assert result["status"]=="VALID" and result["training_tokens"]==256 and result["train_loss"] < 5
    assert (tmp_path/"artifacts/final_recipe_experiments/smoke/milestone.pt").exists()
