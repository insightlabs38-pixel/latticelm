import torch

from latticelm.config import LatticeConfig
from latticelm.model import build_model


def test_sllama_inspired_uses_rrhp_and_shared_block() -> None:
    config = LatticeConfig(
        architecture="sllama_inspired", vocab_size=128, d_model=64,
        n_layers=3, n_heads=4, n_kv_heads=1, ffn_hidden=192,
        context_length=16, tie_embeddings=False,
    )
    model = build_model(config)
    assert model.token_embedding.embedding_dim == 16
    assert len({id(model.block) for _ in range(config.n_layers)}) == 1
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_cosine_optimizer_options_validate() -> None:
    config = LatticeConfig(lr_schedule="cosine", adam_beta1=0.85, adam_beta2=0.98)
    assert config.lr_schedule == "cosine"
