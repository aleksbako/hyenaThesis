import torch
import math
from .OptimModule import OptimModule

class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs): 
        """Complex exponential positional embeddings for Hyena filters."""  
        super().__init__()
        
        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2            
        # To compute the right embeddings we use the "proper" linspace 
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1 
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None] 
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb) 
        self.register("t", t, lr=0.0)
        
    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]
    