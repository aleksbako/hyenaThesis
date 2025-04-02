import torch
import torch.nn as nn
from .HyenaOperator import HyenaOperator
class SE_Block_Hyena(nn.Module):
    def __init__(self, c, l_max, r=16, filter_order=32):
        super(SE_Block_Hyena, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # Using HyenaOperator for excitation part
        self.excitation = HyenaOperator(
            d_model=c,  # The number of input channels (same as input and output)
            l_max=l_max,  # Max sequence length (here, can be the channel dimension)
            order=2,  # Depth of Hyena recurrence (can be tuned)
            filter_order=filter_order,  # Filter order (can be tuned)
            dropout=0.3,
            filter_dropout=0.3
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        # Squeeze step: Global pooling to reduce spatial dimensions
        y = self.squeeze(x).view(bs, 1, c)  # Squeezed tensor (1 channel per feature map)
        
        # Excitation step: Apply Hyena operator on the channel dimension
        y = self.excitation(y).view(bs, c, 1, 1)
        
        # Recalibrate channels
        return x * y.expand_as(x)