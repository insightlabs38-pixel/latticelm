"""Scientific correctness gates for the pinned IHMS causal adaptation."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from latticelm.config import LatticeConfig
from latticelm.full_co4 import FullCo4LM, causal_production
from latticelm.full_co4_reference import causal_reference, released_prefix


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('seed',[17,2026])
@pytest.mark.parametrize('differential',[False,True])
def test_reference_forward_gradient_optimizer_and_selection(seed,differential):
    torch.manual_seed(seed)
    raw=[torch.randn(2,15,8,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    raw += [torch.randn(2,15,1,8,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    copies=[x.detach().clone().requires_grad_() for x in raw]
    lam=torch.tensor(.8001,dtype=torch.float64,requires_grad=True) if differential else None
    lam_copy=lam.detach().clone().requires_grad_() if differential else None
    expected=causal_reference(*raw,top_k=12,differential_lambda=lam)
    actual=causal_production(*copies,top_k=12,chunk_size=4,return_indices=True,differential_lambda=lam_copy)
    for a,b in zip(expected[:2],actual[:2]):torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)
    for t,(reference,production) in enumerate(zip(expected[2],actual[2])):
        for a,b in zip(reference,production):
            assert torch.equal(a,b)
            assert int(b.max())<=t
    sum(x.square().mean() for x in expected[:2]).backward()
    sum(x.square().mean() for x in actual[:2]).backward()
    if differential:raw.append(lam);copies.append(lam_copy)
    for a,b in zip(raw,copies):torch.testing.assert_close(a.grad,b.grad,atol=2e-11,rtol=2e-11)
    opt1=torch.optim.AdamW(raw,lr=1e-4);opt2=torch.optim.AdamW(copies,lr=1e-4)
    opt1.step();opt2.step()
    for a,b in zip(raw,copies):torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-11)


@pytest.mark.parametrize('seed',[17,2026])
@pytest.mark.parametrize('hybrid',[False,True])
@pytest.mark.parametrize('differential',[False,True])
def test_prefix_future_and_gradient_leakage(seed,hybrid,differential):
    torch.manual_seed(seed)
    cfg=LatticeConfig(vocab_size=32,d_model=16,n_layers=2,n_heads=2,n_kv_heads=2,ffn_hidden=32,context_length=16,dropout=0.)
    model=FullCo4LM(cfg,block_types=['modulated','standard'] if hybrid else None,chunk_size=4,differential=differential).double().eval()
    tokens=torch.randint(0,32,(2,15));full=model(tokens)[0]
    for t in [1,7,12]:
        torch.testing.assert_close(model(tokens[:,:t])[0],full[:,:t],atol=2e-11,rtol=2e-11)
        changed=tokens.clone();changed[:,t:]=torch.randint(0,32,changed[:,t:].shape)
        torch.testing.assert_close(model(changed)[0][:,:t],full[:,:t],atol=2e-11,rtol=2e-11)
    captured=[]
    hook=model.token_embedding.register_forward_hook(lambda _,__,output:captured.append(output))
    logits=model(tokens)[0]
    derivative=torch.autograd.grad(logits[:,5,0].sum(),captured[-1])[0]
    hook.remove()
    assert torch.equal(derivative[:,6:],torch.zeros_like(derivative[:,6:]))
    assert torch.isfinite(logits).all()
    assert sum(model.parameter_accounting()[k] for k in ('embedding_head','latent','attention_readout','ffn','normalization'))==sum(p.numel() for p in model.parameters())


def test_compatibility_with_pinned_author_code():
    import hashlib,json
    path=Path(__file__).parent/'fixtures/ihms/attention.py'
    provenance=json.loads((path.parent/'PROVENANCE.json').read_text())
    assert provenance['commit']=='e25e9119173d59fdb9e4548ef640734dbd19f067'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['sha256']
    spec=importlib.util.spec_from_file_location('pinned_ihms_attention',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(123)
    author=module.ModulatedMultiHeadAttention(dim=8,num_latents=1,num_heads=1,dropout=0.).double().eval()
    x=torch.randn(2,17,8,dtype=torch.float64,requires_grad=True)
    q,k,v=author.qkv(x).chunk(3,-1)
    oracle,_=released_prefix(q[:,1:],k[:,1:],v[:,1:],q[:,:1],k[:,:1],v[:,:1])
    output=author(x)
    torch.testing.assert_close(oracle,output,atol=2e-12,rtol=2e-12)
    ga=torch.autograd.grad(oracle.square().sum(),(x,author.qkv.weight),retain_graph=True)
    gb=torch.autograd.grad(output.square().sum(),(x,author.qkv.weight))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=2e-10,rtol=2e-10)
