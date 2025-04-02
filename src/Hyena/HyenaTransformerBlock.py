import torch.nn as nn
from .HyenaOperator import HyenaOperator


def initialize_mlp(mlp):
    for name, param in mlp.named_parameters():
        if param.requires_grad:
            if param.dim() > 1:  # Weight matrices
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')  # Approximates GELU
            elif param.dim() == 1:  # Biases
                nn.init.zeros_(param)
class HyenaTransformerBlock(nn.Module):
    def __init__(self, d_model, l_max, mlp_dim, dropout=0.4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(d_model, l_max, dropout=dropout, filter_dropout=dropout,filter_order=128)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
        initialize_mlp(self.mlp)

    def forward(self, x):
        x = x + self.hyena(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
