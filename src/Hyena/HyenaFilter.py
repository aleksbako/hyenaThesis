import torch
import torch.nn as nn
from .util.OptimModule import OptimModule
from .util.Sin import Sin
from .util.PositionalEmbedding import PositionalEmbedding
from .util.ExponentialModulation import ExponentialModulation

class HyenaFilter(OptimModule):
    def __init__(
            self, 
            d_model,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP 
            fused_fft_conv=False,
            seq_len=1024, 
            lr=1e-3, 
            lr_pos_emb=1e-5,
            dropout=0.0, 
            w=1, # frequency of periodic activations 
            wd=0, # weight decay of kernel parameters 
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.
        
        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)
        
        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
            
        self.modulation = ExponentialModulation(d_model, **kwargs)
        
        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():        
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)
        
        # Ensure compatibility with filters that return a tuple 
        k = k[0] if type(k) is tuple else k 

        y = fftconv(x, k, bias)
        return y
    


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script 
def mul_sum(q, y):
    return (q * y).sum(dim=1)