import torch
import torch.nn as nn
import torch.nn.functional as F

class ModulatedMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with latent token modulation and learnable aggregation.
    """
    def __init__(self, dim, num_latents, num_heads=8, qkv_bias=True, dropout=0.0,
                 modulate_v=True, aggregation='max'):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        assert aggregation in ['mean', 'max', 'learned_weight', 'attention', 'gated'], \
            f"aggregation must be one of ['mean', 'max', 'learned_weight', 'attention', 'gated'], got {aggregation}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.modulate_v = modulate_v
        self.aggregation = aggregation
        self.num_latents = num_latents
        
        # --- VISUALIZATION STATE ---
        self.save_indices = False
        self.saved_indices = None
        # ---------------------------
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(dropout)
    
    def aggregate_tokens(self, tokens):
        if self.aggregation == 'mean':
            return tokens.mean(dim=2, keepdim=True)
        elif self.aggregation == 'max':
            max_tokens, indices = tokens.topk(12, dim=3) # hardcoded 12 for imagenet-1k
            return max_tokens, indices
        elif self.aggregation == 'learned_weight':
            weights = self.aggregation_weight(tokens)
            weights = F.softmax(weights, dim=2)
            weighted = tokens * weights
            return weighted.sum(dim=2, keepdim=True)
        elif self.aggregation == 'attention':
            query = tokens.mean(dim=2, keepdim=True)
            query = self.agg_query(query)
            scores = (query @ tokens.transpose(-2, -1)) * self.agg_scale
            attn_weights = F.softmax(scores, dim=-1)
            aggregated = attn_weights @ tokens
            return aggregated
        elif self.aggregation == 'gated':
            gates = self.gate_net(tokens)
            gated = tokens * gates
            return gated.mean(dim=2, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def modulate_cls_token(self, cls_token, other_tokens, same_tokens):
        cls_token = cls_token.unsqueeze(3)
        other_tokens = other_tokens.unsqueeze(2)
        same_tokens = same_tokens.unsqueeze(2)
        interaction = cls_token * other_tokens # Qcls X Kx
        modulated = interaction + same_tokens  # Qx + (Qcls X Kx)
        modulated_cls, indices = self.aggregate_tokens(modulated)
        return modulated_cls.squeeze(2), indices.squeeze(2)
    
    def modulate_v_cls(self, v_cls ,qk_cls, v_full):
        v_squared = v_full ** 2
        two_v = 2 * v_full
        # product of original qcls and kcls are used here, mean of modulated q_cls and k_cls gives slightly better performance
        qk_interaction = qk_cls 
        v_cls_abs = torch.abs(v_cls)
        gating = 1 + v_cls_abs 
        interaction_term = qk_interaction * gating 
        interaction_term = interaction_term.unsqueeze(3)
        v_squared = v_squared.unsqueeze(2)
        two_v = two_v.unsqueeze(2)
        modulated_v = v_squared + two_v + interaction_term
        v_cls_modulated, indices = self.aggregate_tokens(modulated_v)
        return v_cls_modulated.squeeze(2), indices.squeeze(2), modulated_v.squeeze(2)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        L = self.num_latents 
        q_cls, q_patches = q[:, :, 0:L, :], q[:, :, L:, :]
        k_cls, k_patches = k[:, :, 0:L, :], k[:, :, L:, :]
        v_cls, v_patches = v[:, :, 0:L, :], v[:, :, L:, :]
    
        q_cls_modulated, _ = self.modulate_cls_token(q_cls, k_patches, q_patches)
        k_cls_modulated, _ = self.modulate_cls_token(k_cls, q_patches, k_patches)
        v_cls_modulated, v_indices, modulated_v = self.modulate_v_cls(v_cls, q_cls*k_cls, v_patches)

        # --- CAPTURE INDICES (Added for Visualization) ---
        if self.save_indices:
            # v_indices shape is likely (B, H, 12, D)
            self.saved_indices = v_indices.detach().clone()
        # -------------------------------------------------

        q_cls_modulated = torch.cat([q_cls, q_cls_modulated], dim=2)
        k_cls_modulated = torch.cat([k_cls, k_cls_modulated], dim=2)
        v_cls_modulated = torch.cat([v_cls, v_cls_modulated], dim=2)
    
        attn = (q_cls_modulated @ k_cls_modulated.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
    
        out = attn @ v_cls_modulated

        # --- Reconstructing output to match shapes for scatter ---
        out = out.transpose(1, 2).reshape(B, 13, C) 
        out_cls, out_ = out[:, 0], out[:, 1:]  
        
        # Create zeros tensor with same dtype as out_ to avoid AMP dtype mismatch
        skip_v_zeros = torch.zeros_like(modulated_v.squeeze(1), dtype=out_.dtype)  
        
        # v_indices.squeeze(1) is (B, 12, D). out_ is (B, 12, C). C=D. 
        # This scatter maps the Top K features back to their original token positions (dim=1)
        skip_v_zeros = skip_v_zeros.scatter_(dim=1, index=v_indices.squeeze(1), src=out_)

        v_out = modulated_v.squeeze(1).to(out_.dtype) + skip_v_zeros
        v_out = torch.cat([out_cls.unsqueeze(1), v_out], dim=1)
        
        return v_out


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention using PyTorch's built-in scaled_dot_product_attention.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Use PyTorch's built-in scaled dot product attention
        # This automatically handles scaling, softmax, and optional dropout
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
