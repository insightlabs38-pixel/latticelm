from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def wsd_lr_scale(step: int, total_steps: int, warmup: float = .02,
                 stable: float = .83, decay: float = .15) -> float:
    """Exact 2/83/15 warmup-stable-cosine-decay schedule."""
    if total_steps < 1 or not 1 <= step <= total_steps:
        raise ValueError("step must be in [1, total_steps]")
    warm_steps = max(1, round(total_steps * warmup))
    decay_steps = max(1, round(total_steps * decay))
    decay_start = total_steps - decay_steps + 1
    if step <= warm_steps:
        return step / warm_steps
    if step < decay_start:
        return 1.0
    progress = (step - decay_start + 1) / decay_steps
    return .5 * (1 + math.cos(math.pi * progress))


def muon_parameter_partition(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], dict[str, str]]:
    """Muon for hidden 2-D matrices; AdamW for embeddings/head/norms/latents."""
    muon, adam, names = [], [], {}
    for name, parameter in model.named_parameters():
        unsuitable = (parameter.ndim != 2 or "token_embedding" in name or
                      "lm_head" in name or "norm" in name or "latent_" in name)
        group = adam if unsuitable else muon
        group.append(parameter); names[name] = "adamw" if unsuitable else "muon"
    ids = [id(x) for x in muon + adam]
    expected = [id(x) for x in model.parameters() if x.requires_grad]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise RuntimeError("optimizer partition is not an exact parameter partition")
    return muon, adam, names


class OptimizerBundle:
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers: optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers: optimizer.step()

    def state_dict(self) -> list[dict]:
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, states: list[dict]) -> None:
        if len(states) != len(self.optimizers): raise ValueError("optimizer count mismatch")
        for optimizer, state in zip(self.optimizers, states): optimizer.load_state_dict(state)

    @property
    def param_groups(self) -> list[dict]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]


def make_optimizer(model: torch.nn.Module, kind: str, lr: float, weight_decay: float,
                   betas: tuple[float, float]) -> tuple[OptimizerBundle, dict[str, str]]:
    if kind == "adamw":
        return OptimizerBundle([torch.optim.AdamW(model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=betas, eps=1e-8)]), {
                name: "adamw" for name, _ in model.named_parameters()}
    if kind != "muon_hybrid": raise ValueError(kind)
    muon, adam, names = muon_parameter_partition(model)
    # Muon's scale is deliberately independent from the AdamW peak-LR sweep.
    return OptimizerBundle([
        torch.optim.Muon(muon, lr=.02, momentum=.95, weight_decay=weight_decay,
                         adjust_lr_fn="match_rms_adamw"),
        torch.optim.AdamW(adam, lr=lr, weight_decay=weight_decay,
                          betas=betas, eps=1e-8),
    ]), names
