"""Bounded architectural variants for the post-hackathon campaign.

This module is separate from the frozen production model path. Every variant
uses an explicit factory call so historical checkpoints keep their semantics.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from .co4_reference import CausalCo4Attention, rms_unit
from .config import LatticeConfig
from .model import LatticeLM, SwiGLU, apply_rope


class ValueResidualCo4Attention(CausalCo4Attention):
    """Mix first-layer and current post-MOD values before causal SDPA."""

    def __init__(self, config: LatticeConfig, *, first_layer: bool) -> None:
        if not config.real_gqa:
            raise ValueError("value residual screen requires real grouped-query projections")
        super().__init__(config)
        self.first_layer = first_layer
        if not first_layer:
            self.mix_first = nn.Parameter(torch.tensor(0.5))
            self.mix_current = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, first_value: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = self.mod(self.latent_q, q)
        k = self.mod(self.latent_k, k)
        v = self.mod(self.latent_v, v)
        if self.first_layer:
            if first_value is not None:
                raise ValueError("first layer must establish the residual value")
            first_value = v
        else:
            if first_value is None or first_value.shape != v.shape:
                raise ValueError("first-layer post-MOD value shape mismatch")
            v = self.mix_first * first_value + self.mix_current * v
        if self.qk_norm:
            q, k = rms_unit(q), rms_unit(k)
        q, k = apply_rope(q), apply_rope(k)
        repeat = self.heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        output = self.output(result.transpose(1, 2).contiguous().view(batch, length, -1))
        return output, first_value


class ValueResidualCo4LM(LatticeLM):
    def __init__(self, config: LatticeConfig) -> None:
        if config.architecture != "co4_causal" or config.mixer_strategy != "attention":
            raise ValueError("value residual is defined for causal Co4 attention only")
        super().__init__(config)
        for index, block in enumerate(self.blocks):
            old = block.attn
            new = ValueResidualCo4Attention(config, first_layer=index == 0)
            incompatible = new.load_state_dict(old.state_dict(), strict=False)
            if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"mix_first", "mix_current"}:
                raise RuntimeError("value residual backbone copy mismatch")
            block.attn = new

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(tokens)
        first_value = None
        for block in self.blocks:
            attention_output, first_value = block.attn(block.attn_norm(x), first_value)
            x = x + attention_output
            x = x + block.ffn(block.ffn_norm(x))
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss


class PHMLinear(nn.Module):
    """Learned PHM-4 sum of Kronecker products, with explicit channel ordering."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if in_features % 4 or out_features % 4:
            raise ValueError("PHM-4 requires input and output widths divisible by four")
        self.in_features = in_features
        self.out_features = out_features
        self.algebra = nn.Parameter(torch.empty(4, 4, 4))
        self.blocks = nn.Parameter(torch.empty(4, out_features // 4, in_features // 4))
        nn.init.normal_(self.algebra, std=0.5)
        nn.init.normal_(self.blocks, std=0.02)

    def materialized_weight(self) -> torch.Tensor:
        return sum(torch.kron(self.algebra[index], self.blocks[index]) for index in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.materialized_weight())


class PHMSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.in_proj = PHMLinear(dim, hidden * 2)
        self.out_proj = PHMLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class PHM4FFNCo4LM(LatticeLM):
    """Co4 with PHM-4 only in SwiGLU FFN projections."""

    def __init__(self, config: LatticeConfig) -> None:
        if config.architecture != "co4_causal" or config.mixer_strategy != "attention":
            raise ValueError("PHM-4 FFN screen requires causal Co4 attention")
        super().__init__(config)
        for block in self.blocks:
            block.ffn = PHMSwiGLU(config.d_model, config.ffn_hidden)


class ProgressiveDepthCo4LM(LatticeLM):
    """Dormant, zero-gated Co4 blocks activated at registered growth events."""

    def __init__(self, config: LatticeConfig) -> None:
        if config.architecture != "co4_causal" or config.mixer_strategy != "attention":
            raise ValueError("progressive depth requires causal Co4 attention")
        super().__init__(config)
        self.initial_depth = max(1, config.n_layers // 2)
        self.register_buffer("active_depth", torch.tensor(self.initial_depth, dtype=torch.int64))
        self.growth_gates = nn.ParameterList(
            nn.Parameter(torch.zeros(())) for _ in range(config.n_layers - self.initial_depth))

    def grow(self, new_depth: int) -> None:
        """Clone the last active block into new blocks with zero output gates."""
        old_depth = int(self.active_depth)
        if not old_depth < new_depth <= len(self.blocks):
            raise ValueError("growth must increase active depth within the final model")
        with torch.no_grad():
            for index in range(old_depth, new_depth):
                self.blocks[index].load_state_dict(self.blocks[index - 1].state_dict())
                self.growth_gates[index - self.initial_depth].zero_()
            self.active_depth.fill_(new_depth)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(tokens)
        for index in range(int(self.active_depth)):
            block = self.blocks[index]
            if index < self.initial_depth:
                x = block(x)
            else:
                gate = self.growth_gates[index - self.initial_depth]
                x = x + gate * block.attn(block.attn_norm(x))
                x = x + gate * block.ffn(block.ffn_norm(x))
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss


class TopOneExpertFFN(nn.Module):
    """Sparse top-one FFN with a straight-through router and explicit birth map."""

    def __init__(self, dim: int, hidden: int, initial_experts: int, layer_index: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(SwiGLU(dim, hidden) for _ in range(8))
        self.router = nn.Linear(dim, 8, bias=True)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)
        self.register_buffer("active_experts", torch.tensor(initial_experts, dtype=torch.int64))
        self.register_buffer("clone_map", torch.full((8,), -1, dtype=torch.int64))
        self.register_buffer("cumulative_domain_counts", torch.zeros(4, 8, dtype=torch.int64))
        self.layer_index = layer_index
        self.domain_ids: torch.Tensor | None = None
        self.specialty_catchup = False
        self.forced_expert: int | None = None
        self.ablate_expert: int | None = None
        self.swap_clone_routes = False
        self.last_counts = torch.zeros(8, dtype=torch.int64)
        self.last_domain_counts = torch.zeros(4, 8, dtype=torch.int64)
        self.last_entropy = 0.0
        self.last_aux = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        flat = x.reshape(-1, width)
        active = int(self.active_experts)
        logits = self.router(flat)[:, :active]
        probabilities = torch.softmax(logits, dim=-1)
        selected = logits.argmax(dim=-1)
        if active == 8 and bool((self.clone_map >= 0).any()):
            # On exact cloning, alternation halves routing mass while preserving
            # the forward function. It remains a registered, deterministic rule.
            parity = ((torch.arange(len(selected), device=x.device) + self.layer_index) % 2 == 1)
            child = self.clone_map[selected]
            selected = torch.where(parity & (child >= 0), child, selected)
        if self.specialty_catchup and self.domain_ids is not None and active == 8:
            domain = self.domain_ids.repeat_interleave(length)
            specialty = domain == 3  # FineMath is the registered transition domain.
            child = self.clone_map[selected]
            selected = torch.where(specialty & (child >= 0), child, selected)
        if self.swap_clone_routes and active == 8:
            original = selected.clone()
            for parent in range(4):
                child = int(self.clone_map[parent])
                if child >= 0:
                    selected = torch.where(original == parent, child, selected)
                    selected = torch.where(original == child, parent, selected)
        if self.ablate_expert is not None:
            selected = torch.where(selected == self.ablate_expert,
                                   torch.full_like(selected, (self.ablate_expert + 1) % active),
                                   selected)
        if self.forced_expert is not None:
            if not 0 <= self.forced_expert < active:
                raise ValueError("forced expert must be active")
            selected = torch.full_like(selected, self.forced_expert)
        selected_probability = probabilities.gather(1, selected[:, None]).squeeze(1)
        gate = 1 + selected_probability - selected_probability.detach()
        result = torch.zeros_like(flat)
        for expert_id in range(active):
            mask = selected == expert_id
            if bool(mask.any()):
                result[mask] = self.experts[expert_id](flat[mask]) * gate[mask, None]
        counts = torch.bincount(selected.detach().cpu(), minlength=8)
        self.last_counts = counts
        if self.domain_ids is not None:
            domain = self.domain_ids.repeat_interleave(length).detach().cpu()
            self.last_domain_counts = torch.bincount(domain * 8 + selected.detach().cpu(),
                                                      minlength=32).reshape(4, 8)
            self.cumulative_domain_counts += self.last_domain_counts
        else:
            self.last_domain_counts = torch.zeros(4, 8, dtype=torch.int64)
        self.last_entropy = float((-(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1)
                                   .mean()).detach())
        fraction = counts[:active].to(probabilities.device, dtype=probabilities.dtype) / len(selected)
        self.last_aux = .01 * active * (probabilities.mean(0) * fraction).sum()
        return result.reshape(batch, length, width)

    def birth(self, strategy: str, preferred_parent: int) -> None:
        if int(self.active_experts) != 4 or strategy not in {"random", "clone"}:
            raise ValueError("expert birth requires four initial experts and a registered strategy")
        if strategy == "clone":
            parents = [preferred_parent] + [i for i in range(4) if i != preferred_parent]
            with torch.no_grad():
                for child, parent in enumerate(parents, start=4):
                    self.experts[child].load_state_dict(self.experts[parent].state_dict())
                    self.router.weight[child].copy_(self.router.weight[parent])
                    self.router.bias[child].copy_(self.router.bias[parent])
                    self.clone_map[parent] = child
        self.active_experts.fill_(8)


class MoECo4LM(LatticeLM):
    """Eight-expert total capacity with one active FFN per token and block."""

    DOMAIN_IDS = {"fineweb_edu": 0, "fineweb": 1, "wikipedia": 2, "finemath": 3}

    def __init__(self, config: LatticeConfig, initial_experts: int) -> None:
        if config.architecture != "co4_causal" or initial_experts not in (4, 8):
            raise ValueError("MoE pilot requires causal Co4 and four/eight initial experts")
        super().__init__(config)
        for index, block in enumerate(self.blocks):
            block.ffn = TopOneExpertFFN(config.d_model, config.ffn_hidden,
                                        initial_experts, index)
        self.domain_ids: torch.Tensor | None = None

    def set_domains(self, names: list[str] | None) -> None:
        self.domain_ids = (torch.tensor([self.DOMAIN_IDS[name] for name in names], dtype=torch.int64)
                           if names is not None else None)

    def birth(self, strategy: str) -> dict:
        parents = []
        for block in self.blocks:
            # Pre-registered parent: expert with most FineMath token assignments;
            # ties resolve to the lower index.
            counts = block.ffn.cumulative_domain_counts[3, :4]
            parent = int(counts.argmax())
            parents.append(parent)
            block.ffn.birth(strategy, parent)
        return {"strategy": strategy, "preferred_parents_by_layer": parents}

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = x + block.attn(block.attn_norm(x))
            block.ffn.domain_ids = self.domain_ids if self.domain_ids is not None and len(self.domain_ids) == len(tokens) else None
            x = x + block.ffn(block.ffn_norm(x))
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    def routing_diagnostics(self) -> dict:
        return {"counts_by_layer": [block.ffn.last_counts.tolist() for block in self.blocks],
                "domain_counts_by_layer": [block.ffn.last_domain_counts.tolist() for block in self.blocks],
                "router_entropy_by_layer": [block.ffn.last_entropy for block in self.blocks],
                "load_balance_loss": float(sum(block.ffn.last_aux.detach() for block in self.blocks))}

    def auxiliary_loss(self) -> torch.Tensor:
        return sum(block.ffn.last_aux for block in self.blocks)

    def set_catchup(self, enabled: bool) -> None:
        for block in self.blocks:
            block.ffn.specialty_catchup = enabled

    def active_parameter_count(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dormant = sum(sum(p.numel() for p in expert.parameters())
                      for block in self.blocks for expert in block.ffn.experts[1:])
        return total - dormant


class NormalizedCo4Attention(CausalCo4Attention):
    """Co4 MOD with unit-direction projections and learned Q/K logit scale."""

    def __init__(self, config: LatticeConfig) -> None:
        super().__init__(config)
        self.log_qk_scale = nn.Parameter(torch.full((self.heads,), math.log(self.head_dim)))

    @staticmethod
    def linear(x: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        return F.linear(x, F.normalize(module.weight, dim=1, eps=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.linear(x, self.q_proj).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.linear(x, self.k_proj).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.linear(x, self.v_proj).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = self.mod(self.latent_q, q)
        k = self.mod(self.latent_k, k)
        v = self.mod(self.latent_v, v)
        q = F.normalize(q, dim=-1, eps=1e-6) * self.log_qk_scale.exp()[None, :, None, None]
        k = F.normalize(k, dim=-1, eps=1e-6)
        q, k = apply_rope(q), apply_rope(k)
        repeat = self.heads // self.kv_heads
        result = F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, 1),
                                                 v.repeat_interleave(repeat, 1),
                                                 is_causal=True, dropout_p=0.0)
        return self.linear(result.transpose(1, 2).contiguous().view(batch, length, -1),
                           self.output)


class HypersphericalCo4LM(LatticeLM):
    """Bounded nGPT-style pilot with functional normalization and learned steps."""

    def __init__(self, config: LatticeConfig) -> None:
        if config.architecture != "co4_causal" or not config.real_gqa:
            raise ValueError("nCo4 pilot requires causal Co4 with real GQA")
        super().__init__(config)
        self.attn_step_logits = nn.Parameter(torch.full((config.n_layers,), math.log(.1 / .9)))
        self.ffn_step_logits = nn.Parameter(torch.full((config.n_layers,), math.log(.1 / .9)))
        self.ffn_log_gains = nn.Parameter(torch.zeros(config.n_layers))
        self.logit_log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(config.d_model))))
        for block in self.blocks:
            old = block.attn
            new = NormalizedCo4Attention(config)
            incompatible = new.load_state_dict(old.state_dict(), strict=False)
            if incompatible.unexpected_keys or set(incompatible.missing_keys) != {"log_qk_scale"}:
                raise RuntimeError("normalized attention backbone copy mismatch")
            block.attn = new

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = F.normalize(self.token_embedding.weight, dim=-1, eps=1e-6)
        x = F.normalize(F.embedding(tokens, embedding), dim=-1, eps=1e-6)
        for index, block in enumerate(self.blocks):
            attn = F.normalize(block.attn(x), dim=-1, eps=1e-6)
            step = torch.sigmoid(self.attn_step_logits[index])
            x = F.normalize((1 - step) * x + step * attn, dim=-1, eps=1e-6)
            ffn_input = F.linear(x, F.normalize(block.ffn.in_proj.weight, dim=1, eps=1e-6))
            value, gate = ffn_input.chunk(2, dim=-1)
            hidden = F.silu(gate) * value
            ffn = F.linear(hidden, F.normalize(block.ffn.out_proj.weight, dim=1, eps=1e-6))
            ffn = F.normalize(ffn, dim=-1, eps=1e-6) * self.ffn_log_gains[index].exp()
            step = torch.sigmoid(self.ffn_step_logits[index])
            x = F.normalize((1 - step) * x + step * ffn, dim=-1, eps=1e-6)
        head = F.normalize(self.lm_head.weight, dim=-1, eps=1e-6)
        self.last_hidden_norm_max_error = float((x.detach().norm(dim=-1) - 1).abs().max())
        logits = F.linear(x, head) * self.logit_log_scale.exp()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss


class DifferentialCo4Attention(CausalCo4Attention):
    """Two causal maps from one post-MOD Q/K/V projection with learned lambda."""

    def __init__(self, config: LatticeConfig) -> None:
        if not config.real_gqa or config.d_model // config.n_heads < 4:
            raise ValueError("Differential-Co4 requires real GQA and head width >=4")
        super().__init__(config)
        first = self.head_dim // 2
        second = self.head_dim - first
        self.lambda_q1 = nn.Parameter(torch.randn(first) * .02)
        self.lambda_k1 = nn.Parameter(torch.randn(first) * .02)
        self.lambda_q2 = nn.Parameter(torch.randn(second) * .02)
        self.lambda_k2 = nn.Parameter(torch.randn(second) * .02)
        self.lambda_init = .8
        self.disable_differential = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable_differential:
            return super().forward(x)
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = self.mod(self.latent_q, q)
        k = self.mod(self.latent_k, k)
        v = self.mod(self.latent_v, v)
        if self.qk_norm:
            q, k = rms_unit(q), rms_unit(k)
        q, k = apply_rope(q), apply_rope(k)
        k = k.repeat_interleave(self.heads // self.kv_heads, 1)
        v = v.repeat_interleave(self.heads // self.kv_heads, 1)
        cut = self.head_dim // 2
        first = F.scaled_dot_product_attention(q[..., :cut], k[..., :cut], v,
                                                is_causal=True, dropout_p=0.0)
        second = F.scaled_dot_product_attention(q[..., cut:], k[..., cut:], v,
                                                 is_causal=True, dropout_p=0.0)
        lam = (torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) -
               torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) + self.lambda_init)
        mixed = (first - lam * second) / (1 - self.lambda_init)
        return self.output(mixed.transpose(1, 2).contiguous().view(batch, length, -1))


class DifferentialCo4LM(LatticeLM):
    def __init__(self, config: LatticeConfig) -> None:
        if config.architecture != "co4_causal":
            raise ValueError("Differential-Co4 requires causal Co4")
        super().__init__(config)
        for block in self.blocks:
            old = block.attn
            new = DifferentialCo4Attention(config)
            incompatible = new.load_state_dict(old.state_dict(), strict=False)
            expected = {"lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"}
            if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected:
                raise RuntimeError("differential attention backbone copy mismatch")
            block.attn = new


def build_posthack_model(config: LatticeConfig, variant: str = "baseline") -> nn.Module:
    if variant == "baseline":
        from .model import build_model
        return build_model(config)
    if variant == "value_residual":
        return ValueResidualCo4LM(config)
    if variant == "phm4_ffn":
        return PHM4FFNCo4LM(config)
    if variant == "progressive_depth":
        return ProgressiveDepthCo4LM(config)
    if variant == "moe_fixed":
        return MoECo4LM(config, 8)
    if variant == "moe_growth":
        return MoECo4LM(config, 4)
    if variant == "nco4":
        return HypersphericalCo4LM(config)
    if variant == "differential_co4":
        return DifferentialCo4LM(config)
    raise ValueError(f"unknown post-hackathon model variant: {variant}")
