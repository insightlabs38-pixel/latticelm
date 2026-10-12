"""Endpoint counterfactuals for the developmental-expert mechanism."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.posthack_variants import MoECo4LM, build_posthack_model
from posthack_campaign import ROOT, atomic_json, sha256, utc
from train_final_production import load_arrays

MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"


def batches(valid: dict, context: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for source, arrays in valid.items():
        values = np.concatenate(arrays)
        chunks = [np.asarray(values[i*context:(i+1)*context+1]) for i in range(2)]
        result[source] = (torch.tensor(np.stack([part[:-1] for part in chunks]), dtype=torch.long),
                          torch.tensor(np.stack([part[1:] for part in chunks]), dtype=torch.long))
    return result


def loss(model: MoECo4LM, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.inference_mode():
        logits, _ = model(x)
        return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))


def control(model: MoECo4LM, name: str, value) -> None:
    for block in model.blocks:
        setattr(block.ffn, name, value)


def diagnose(checkpoint_path, output) -> dict:
    checkpoint_path = Path(checkpoint_path)
    output = Path(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = LatticeConfig(**checkpoint["config"])
    if checkpoint.get("variant") not in ("moe_fixed", "moe_growth"):
        raise ValueError("endpoint is not a developmental-expert model")
    torch.set_num_threads(4)
    model = build_posthack_model(cfg, checkpoint["variant"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _, _, valid = load_arrays(MANIFEST)
    data = batches(valid, cfg.context_length)
    baseline = {source: loss(model, *pair) for source, pair in data.items()}
    forced, ablated = {}, {}
    for expert_id in range(8):
        control(model, "forced_expert", expert_id)
        forced[str(expert_id)] = {source: loss(model, *pair) for source, pair in data.items()}
        control(model, "forced_expert", None)
        control(model, "ablate_expert", expert_id)
        ablated[str(expert_id)] = {source: loss(model, *pair) - baseline[source]
                                    for source, pair in data.items()}
        control(model, "ablate_expert", None)
    control(model, "swap_clone_routes", True)
    swapped = {source: loss(model, *pair) - baseline[source] for source, pair in data.items()}
    control(model, "swap_clone_routes", False)
    captured = {}
    hooks = [block.ffn_norm.register_forward_hook(
        lambda _module, _input, output, layer=index:
        captured.__setitem__(layer, output.detach().reshape(-1, cfg.d_model)[:128]))
        for index, block in enumerate(model.blocks)]
    loss(model, *data["finemath"])
    for hook in hooks:
        hook.remove()
    representation = []
    for layer, block in enumerate(model.blocks):
        for parent in range(4):
            child = int(block.ffn.clone_map[parent])
            if child < 0:
                continue
            with torch.inference_mode():
                parent_output = block.ffn.experts[parent](captured[layer])
                child_output = block.ffn.experts[child](captured[layer])
                cosine = float(F.cosine_similarity(parent_output, child_output, dim=-1).mean())
            parent_parameters = torch.cat([p.detach().reshape(-1) for p in block.ffn.experts[parent].parameters()])
            child_parameters = torch.cat([p.detach().reshape(-1) for p in block.ffn.experts[child].parameters()])
            distance = float((child_parameters - parent_parameters).norm() / parent_parameters.norm().clamp_min(1e-9))
            representation.append({"layer": layer, "parent": parent, "child": child,
                                   "output_cosine_on_finemath_activation": cosine,
                                   "relative_parameter_distance": distance})
    gradients = {}
    for source in ("fineweb_edu", "finemath"):
        x, y = data[source]
        model.set_domains([source] * len(x))
        model.zero_grad(set_to_none=True)
        logits, _ = model(x)
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).backward()
        gradient = model.blocks[0].ffn.router.weight.grad
        gradients[source] = gradient.detach().reshape(-1).clone() if gradient is not None else None
    model.set_domains(None)
    if any(value is None for value in gradients.values()):
        conflict = None
    else:
        conflict = float(F.cosine_similarity(gradients["fineweb_edu"],
                                             gradients["finemath"], dim=0))
    report = {"schema": "posthack-moe-endpoint-diagnostics-v1", "created_utc": utc(),
              "checkpoint_sha256": sha256(checkpoint_path),
              "validation_policy": "first two 256-token windows per certified validation source",
              "baseline_loss_by_source": baseline,
              "forced_expert_loss_by_source": forced,
              "expert_ablation_loss_increase_by_source": ablated,
              "parent_child_swap_loss_increase_by_source": swapped,
              "parent_child_representation_and_parameter_distance": representation,
              "fineweb_edu_vs_finemath_first_router_gradient_cosine": conflict,
              "diagnostic_limit": "small fixed windows and first-router gradient estimate; infer mechanism only alongside held-out and broad capability changes"}
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.checkpoint, args.output), indent=2))
