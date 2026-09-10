import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ==============================================================================
#  Time Embedding
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [batch_size, 1]
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ==============================================================================
# Conditional Flow MLP (c = DEG Embedding)
# ==============================================================================
class GatedMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        
        self.norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, inner_dim)
        self.in_proj = nn.Linear(dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        
        gate = F.silu(self.gate_proj(x))
        val = self.in_proj(x)
        x = gate * val
        
        x = self.dropout(x)
        x = self.out_proj(x)
        return identity + x

class GatedConditionalFlowMLP(nn.Module):
    def __init__(self, 
                 embedding_dim: int = 256,
                 condition_dim: int = 256,
                 model_dim: int = 512, 
                 num_layers: int = 6, 
                 combine_method: str = 'sum',
                 dropout: float = 0.1):
        super().__init__()
        self.combine_method = combine_method
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        self.input_proj = nn.Linear(embedding_dim, model_dim)
        
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(condition_dim), 
            nn.Linear(condition_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        if self.combine_method == 'concat':
            self.combine_proj = nn.Linear(model_dim * 3, model_dim)
        elif self.combine_method == 'cross_attn':
            self.attn_norm = nn.LayerNorm(model_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=8, batch_first=True, dropout=dropout)
            self.attn_proj = nn.Linear(model_dim, model_dim)
        
        self.layers = nn.ModuleList([
            GatedMLPBlock(model_dim, expansion_factor=2, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, embedding_dim)
        
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.register_buffer('null_condition', torch.zeros(1, condition_dim))

    def forward(self, x, t, condition):
        t_emb = self.time_embed(t)
        x_proj = self.input_proj(x)
        c_proj = self.cond_proj(condition)
        
        if self.combine_method == 'concat':
            h = self.combine_proj(torch.cat([x_proj, t_emb, c_proj], dim=-1))
        elif self.combine_method == 'sum':
            h = x_proj + t_emb + c_proj
        elif self.combine_method == 'cross_attn':
            query = x_proj + t_emb
            q_in = self.attn_norm(query).unsqueeze(1)
            kv_in = c_proj.unsqueeze(1)
            attn_out, _ = self.cross_attn(query=q_in, key=kv_in, value=kv_in)
            h = query + self.attn_proj(attn_out.squeeze(1))

        for layer in self.layers:
            h = layer(h)
            
        return self.output_proj(self.output_norm(h))
