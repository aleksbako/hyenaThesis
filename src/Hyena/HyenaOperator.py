import torch.nn as nn
import torch
from einops import rearrange
from .HyenaFilter import HyenaFilter

class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2, 
            filter_order=64,
            dropout=0.0,  
            filter_dropout=0.0, 
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.short_filter = nn.Conv1d(
            inner_width, 
            inner_width, 
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), 
            order=filter_order, 
            seq_len=l_max,
            channels=1, 
            dropout=filter_dropout, 
            **filter_args
        ) 

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[...,:l_filter] 
        *x, v = uc.split(self.d_model, dim=1)
        
        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)
        
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y

    
if __name__ == "__main__":
    layer = HyenaOperator(
        d_model=512, 
        l_max=1024, 
        order=2, 
        filter_order=64
    )
    x = torch.randn(1, 1024, 512, requires_grad=True)
    y = layer(x)
        
    print(x.shape, y.shape)
    
    grad = torch.autograd.grad(y[:, 10, :].sum(), x)[0]
    print('Causality check: gradients should not flow "from future to past"')
    print(grad[0, 11, :].sum(), grad[0, 9, :].sum())