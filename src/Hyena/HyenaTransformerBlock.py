import torch.nn as nn
from .HyenaOperator import HyenaOperator

class HyenaTransformerBlock(nn.Module):
    def __init__(self, d_model, l_max, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(d_model, l_max, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.hyena(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
