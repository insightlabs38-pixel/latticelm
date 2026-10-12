from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import LatticeConfig
from .memory import ConditionalMemory, RMSNorm


class Co4InspiredExperimental(nn.Module):
    """Isolated, causal O(T * latent_agents) triadic-latent experiment.

    This is intentionally *not* a Co4 reproduction. The BabyLM paper describes
    non-parametric Q/K/V triadic MOD loops but delegates the MOD transfer
    function to a separate cited work and does not publish sufficient equations
    to reproduce it. This class tests only the high-level idea: one layer, two
    heads, a small latent-agent set and causal triadic co-evolution.
    """
    def __init__(self, config: LatticeConfig, latent_agents: int = 24) -> None:
        super().__init__()
        if config.n_heads != 2:
            raise ValueError("Co4InspiredExperimental requires exactly two heads")
        self.config, self.heads, self.agents = config, config.n_heads, latent_agents
        self.head_dim = config.d_model // config.n_heads
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.initial_agents = nn.Parameter(torch.empty(3, latent_agents, self.heads, self.head_dim))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        nn.init.normal_(self.initial_agents, std=0.02)
        self.memory = ConditionalMemory(config.memory_slots, config.memory_dim, config.d_model) if config.memory_enabled else None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length = tokens.shape
        pos = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(pos)[None]
        if self.memory is not None:
            x = self.memory(tokens, x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (item.view(shape) for item in (q, k, v))
        agents = self.initial_agents[:, None].expand(-1, batch, -1, -1, -1).clone()
        output = []
        scale = self.head_dim ** -0.5
        for t in range(length):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            aq, ak, av = agents
            # Parameter-free triadic modulation: each population is modulated
            # by the other two and current causal token representation.
            aq = torch.tanh(aq + 0.1 * (ak + av) + qt[:, None])
            ak = torch.tanh(ak + 0.1 * (aq + av) + kt[:, None])
            av = torch.tanh(av + 0.1 * (aq + ak) + vt[:, None])
            agents = torch.stack((aq, ak, av))
            scores = (qt[:, None] * ak).sum(-1).transpose(1, 2) * scale
            weights = torch.softmax(scores, dim=-1)
            context = torch.einsum("bha,bahd->bhd", weights, av)
            output.append(context.reshape(batch, -1))
        hidden = self.norm(torch.stack(output, dim=1))
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    def parameter_breakdown(self) -> dict[str, int]:
        memory = self.memory.parameter_count if self.memory is not None else 0
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embeddings = self.token_embedding.weight.numel()
        return {"token_embeddings_tied_head": embeddings, "conditional_memory": memory, "backbone_and_norms": total - embeddings - memory, "total": total}

    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.context_length :])
            tokens = torch.cat((tokens, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
        return tokens
