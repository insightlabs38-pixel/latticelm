"""Source-grounded IHMS causal LM adaptation; no BabyLM reproduction claim.

Attribution: ARIA-Funded-TREND/IHMS e25e9119 (CC-BY-NC-4.0).
Production batches prefixes in bounded chunks; the separate oracle is literal.
Prefix selection is quadratic, not the paper's claimed near-linear algorithm.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


def prefix_topk(projected,allowed,count):
    """Descending value, earliest index for exact boundary ties.

    torch.topk's arbitrary tie choice can depend on padded future length. Keep
    its fast path, then sort only feature/prefix rows with an ambiguous cutoff.
    """
    masked=projected.masked_fill(~allowed[None,:,:,None],float('-inf'))
    values,indices=masked.topk(count,dim=2)
    boundary=values[:,:,-1:,:]
    ambiguous=((masked==boundary).sum(2)>(values==boundary).sum(2)) & (allowed.sum(1)>count)[None,:,None]
    if ambiguous.any():
        rows=masked.permute(0,1,3,2)[ambiguous]
        chosen=rows.argsort(dim=-1,descending=True,stable=True)[:,:count]
        revised=indices.permute(0,1,3,2).clone();revised[ambiguous]=chosen
        indices=revised.permute(0,1,3,2)
        values=masked.gather(2,indices)
    if count>1 and (values[:,:,:-1,:]==values[:,:,1:,:]).any():
        # Fix ordering inside a tied selected set too: although its values are
        # equal, feature-wise gathering can otherwise choose different input
        # subgradients across ranks. This sorts at most k<=12 selected items.
        indices=indices.sort(dim=2).values
        values=masked.gather(2,indices)
        ranks=values.argsort(dim=2,descending=True,stable=True)
        indices=indices.gather(2,ranks);values=values.gather(2,ranks)
    return values,indices


def causal_production(q,k,v,ql,kl,vl,top_k=12,chunk_size=8,return_indices=False,differential_lambda=None):
    # q/k/v: B,T,D; latent projections: B,T,1,D. No future information enters
    # a selector, and only the current prefix's current-token update is emitted.
    batch,length,dim=q.shape
    token_out,latent_out,indices_out=[],[],[]
    for start in range(0,length,chunk_size):
        end=min(length,start+chunk_size)
        positions=torch.arange(start,end,device=q.device)
        allowed=torch.arange(end,device=q.device)[None,:] <= positions[:,None]
        qc,kc,vc=[x[:,start:end] for x in (ql,kl,vl)]
        qp,kp,vp=[x[:,:end].unsqueeze(1) for x in (q,k,v)]
        qm=qp+qc*kp; km=kp+kc*qp
        vm=vp.square()+2*vp+(qc*kc)*(1+vc.abs())
        # Variable prefix lengths share a padded selection set. Padding scores
        # never participate in softmax, including the earliest short prefixes.
        count=min(top_k,end)
        selection=[]; all_indices=[]
        for projected in (qm,km,vm):
            values,idx=prefix_topk(projected,allowed,count)
            selection.append(values); all_indices.append(idx)
        valid_slots=torch.arange(count,device=q.device)[None,:] < (positions+1).clamp(max=count)[:,None]
        selected=[x.masked_fill(~valid_slots[None,:,:,None],0) for x in selection]
        qa,ka,va=[torch.cat([latent,value],dim=2) for latent,value in zip((qc,kc,vc),selected)]
        slots=torch.cat([torch.ones((end-start,1),dtype=torch.bool,device=q.device),valid_slots],1)
        if differential_lambda is None:
            scores=(qa @ ka.transpose(-1,-2))*dim**-.5
            weights=scores.masked_fill(~slots[None,:,None,:],float('-inf')).softmax(-1)
            out=weights @ va
        else:
            cut=dim//2
            first=(qa[...,:cut] @ ka[...,:cut].transpose(-1,-2))*cut**-.5
            second=(qa[...,cut:] @ ka[...,cut:].transpose(-1,-2))*(dim-cut)**-.5
            first=first.masked_fill(~slots[None,:,None,:],float('-inf')).softmax(-1) @ va
            second=second.masked_fill(~slots[None,:,None,:],float('-inf')).softmax(-1) @ va
            out=(first-differential_lambda*second)/.2
        # Source scatter sends each selected channel to its V selection index.
        # We emit only t's update; future prefixes cannot rewrite earlier tokens.
        current=(all_indices[2]==positions[None,:,None,None]) & valid_slots[None,:,:,None]
        update=(out[:,:,1:,:]*current).sum(2)
        last_v=vm[:,torch.arange(end-start,device=q.device),positions,:]
        token_out.append(last_v+update);latent_out.append(out[:,:,:1,:])
        if return_indices:
            for offset,t in enumerate(range(start,end)):
                n=min(top_k,t+1)
                indices_out.append(tuple(x[:,offset,:n] for x in all_indices))
    return torch.cat(token_out,1),torch.cat(latent_out,1),indices_out


class IHMSAttention(nn.Module):
    def __init__(self,dim,top_k=12,chunk_size=8,differential=False):
        super().__init__();self.qkv=nn.Linear(dim,3*dim,bias=True)
        self.top_k=top_k;self.chunk_size=chunk_size
        self.differential=differential
        if differential:
            cut=dim//2
            for name,size in [('lambda_q1',cut),('lambda_k1',cut),('lambda_q2',dim-cut),('lambda_k2',dim-cut)]:
                setattr(self,name,nn.Parameter(torch.randn(size)*.02))

    def forward(self,x,latent):
        q,k,v=self.qkv(x).chunk(3,-1)
        ql,kl,vl=self.qkv(latent).chunk(3,-1)
        lam=None
        if self.differential:lam=torch.exp(torch.dot(self.lambda_q1,self.lambda_k1))-torch.exp(torch.dot(self.lambda_q2,self.lambda_k2))+.8
        output,new_latent,_=causal_production(q,k,v,ql,kl,vl,self.top_k,self.chunk_size,differential_lambda=lam)
        return output,new_latent


class CausalMHSA(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();self.heads=heads;self.qkv=nn.Linear(dim,3*dim);self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        return self.proj(out.transpose(1,2).reshape(b,t,d))


class IHMSBlock(nn.Module):
    def __init__(self,dim,hidden,kind,heads,top_k,chunk_size,differential=False):
        super().__init__();self.kind=kind
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim)
        self.attn=IHMSAttention(dim,top_k,chunk_size,differential) if kind=='modulated' else CausalMHSA(dim,heads)
        self.mlp=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,dim))
    def forward(self,x,latent):
        if self.kind=='modulated':
            update,latent_update=self.attn(self.norm1(x),self.norm1(latent))
            x=x+update;latent=latent+latent_update
            latent=latent+self.mlp(self.norm2(latent))
        else:
            x=x+self.attn(self.norm1(x))
        return x+self.mlp(self.norm2(x)),latent


class FullCo4LM(nn.Module):
    def __init__(self,config,family='ihms_causal',block_types=None,top_k=12,chunk_size=8,
                 initialization_scale=.02,latent_count=1,modulated_heads=1,differential=False):
        super().__init__();self.config=config
        if latent_count!=1 or modulated_heads!=1:raise ValueError('Released IHMS operator is supported with one latent and one head')
        if family not in ('ihms_causal','ihms_hybrid'):raise ValueError('Unsupported source family')
        self.family=family;self.top_k=top_k
        order=block_types or ['modulated']*config.n_layers
        if len(order)!=config.n_layers or any(x not in ('modulated','standard') for x in order):raise ValueError('Invalid block order')
        self.block_types=order
        self.token_embedding=nn.Embedding(config.vocab_size,config.d_model)
        self.token_embedding_positions=nn.Parameter(torch.empty(config.context_length,config.d_model))
        if 'modulated' in order:self.latent_agents=nn.Parameter(torch.empty(1,1,1,config.d_model))
        else:self.register_buffer('latent_agents',torch.zeros(1,1,1,config.d_model))
        self.blocks=nn.ModuleList([IHMSBlock(config.d_model,config.ffn_hidden,kind,config.n_heads,top_k,chunk_size,differential) for kind in order])
        self.norm=nn.LayerNorm(config.d_model);self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.apply(lambda m:self._init(m,initialization_scale))
        nn.init.normal_(self.token_embedding_positions,std=initialization_scale)
        if 'modulated' in order:nn.init.normal_(self.latent_agents,std=initialization_scale)
        if config.tie_embeddings:self.lm_head.weight=self.token_embedding.weight

    @staticmethod
    def _init(module,scale):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=scale)
            if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)

    def forward(self,tokens,targets=None):
        b,t=tokens.shape
        if t>self.config.context_length:raise ValueError('Context exceeded')
        x=self.token_embedding(tokens)+self.token_embedding_positions[:t]
        latent=self.latent_agents.expand(b,t,1,-1)
        for block in self.blocks:x,latent=block(x,latent)
        logits=self.lm_head(self.norm(x))
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
        self.last_token_state=x.detach();self.last_latent_state=latent.detach()
        if self.training:
            self.last_training_token_state=x.detach();self.last_training_latent_state=latent.detach()
        return logits,loss

    def architecture_state(self):
        statistics={}
        for name in ('last_token_state','last_latent_state','last_training_token_state','last_training_latent_state','latent_agents'):
            value=getattr(self,name,None)
            if value is not None:
                statistics[name]={'min':float(value.min()),'max':float(value.max()),'mean':float(value.mean()),
                                  'rms':float(value.square().mean().sqrt()),'abs_gt6_fraction':float((value.abs()>6).float().mean()),
                                  'saturation_fraction':None,'saturation_note':'No bounded activation or ReLU6 in this released random-latent operator.'}
        return {'family':self.family,'block_types':self.block_types,'latent_state':'stateless across batches; prefix-local within each forward',
                'top_k':self.top_k,'activation_statistics_last_validation_forward':statistics,'parameter_accounting':self.parameter_accounting(),
                'sequence_complexity':'quadratic prefix modulation/selection; worst-case tied-selector stable repair O(T*T*D*log(T)); selected attention O(T*k*k*D); no near-linear claim'}

    def parameter_accounting(self):
        groups={'embedding_head':0,'latent':0,'attention_readout':0,'ffn':0,'normalization':0}
        for name,p in self.named_parameters():
            key=('embedding_head' if any(s in name for s in ('token_embedding','position_embedding','lm_head'))
                 else 'latent' if 'latent_agents' in name else 'ffn' if '.mlp.' in name
                 else 'attention_readout' if '.attn.' in name else 'normalization')
            groups[key]+=p.numel()
        groups['total_trainable']=sum(groups.values());groups['active_parameters_per_token']=groups['total_trainable']
        return groups
