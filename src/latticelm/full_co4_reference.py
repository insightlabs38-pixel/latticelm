"""Literal oracle for a causal adaptation of IHMS e25e9119.

Attribution: ARIA-Funded-TREND/IHMS, Vision/object_recognition/attention.py,
CC-BY-NC-4.0. Prefix-local selection/readout is our disclosed LM adaptation.
One latent, one head, source V formula, independent per-feature top-k.
"""
import torch


def released_prefix(q, k, v, ql, kl, vl, top_k=12, differential_lambda=None):
    """Literal released single-prefix operator; returns latent plus every token."""
    qm = q + ql * k
    km = k + kl * q
    vm = v**2 + 2*v + (ql*kl)*(1+vl.abs())
    count = min(top_k, q.shape[-2])
    # Source topk leaves exact ties unspecified. The causal LM adaptation
    # explicitly chooses earliest prefix indices so future padding cannot
    # change either the emitted token update or a tie's chosen subgradient.
    qi = qm.argsort(dim=-2, descending=True, stable=True)[...,:count,:]
    ki = km.argsort(dim=-2, descending=True, stable=True)[...,:count,:]
    vi = vm.argsort(dim=-2, descending=True, stable=True)[...,:count,:]
    qs, ks, vs = qm.gather(-2,qi), km.gather(-2,ki), vm.gather(-2,vi)
    qa = torch.cat([ql, qs], dim=-2)
    ka = torch.cat([kl, ks], dim=-2)
    va = torch.cat([vl, vs], dim=-2)
    if differential_lambda is None:
        weights = (qa @ ka.transpose(-1,-2) * q.shape[-1]**-.5).softmax(-1)
        output = weights @ va
    else:
        cut=qa.shape[-1]//2
        first=(qa[...,:cut] @ ka[...,:cut].transpose(-1,-2)*cut**-.5).softmax(-1) @ va
        second=(qa[...,cut:] @ ka[...,cut:].transpose(-1,-2)*(qa.shape[-1]-cut)**-.5).softmax(-1) @ va
        output=(first-differential_lambda*second)/.2
    updates = torch.zeros_like(vm).scatter(-2, vi, output[...,1:,:])
    return torch.cat([output[...,:1,:], vm+updates], dim=-2), (qi, ki, vi)


def causal_reference(q, k, v, ql, kl, vl, top_k=12, differential_lambda=None):
    tokens, latents, selected = [], [], []
    for t in range(q.shape[1]):
        output, indices = released_prefix(q[:,:t+1], k[:,:t+1], v[:,:t+1],
                                          ql[:,t], kl[:,t], vl[:,t], top_k,differential_lambda)
        tokens.append(output[:,-1]); latents.append(output[:,0]); selected.append(indices)
    return torch.stack(tokens,1), torch.stack(latents,1).unsqueeze(2), selected
