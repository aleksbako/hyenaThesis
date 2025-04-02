
import torch
import torch.nn as nn
import torch.nn.functional as F
from .HyenaOperator import HyenaOperator
class Hyena_AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, order=2, dropout=0.0, filter_order=32, stride=1, filter_dropout=0.2):
        super(Hyena_AugmentedConv, self).__init__()

        # Validate that out_channels is greater than or equal to in_channels
        if out_channels < in_channels:
            raise ValueError("out_channels must be greater than or equal to in_channels.")

        self.hyena = HyenaOperator(
            d_model=in_channels,
            l_max=1024,  # Will adjust based on input spatial dimensions
            order=order,
            dropout=dropout,
            filter_order=filter_order,
            filter_dropout=filter_dropout
        )

        self.conv_out = None
        if out_channels > in_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=stride, padding=1)

        self.proj_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        l_max = H * W

        # Prepare for Hyena
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, l_max, C)
        hyena_out = self.hyena(x_flat)
        hyena_out = hyena_out.view(B, H, W, -1).permute(0, 3, 1, 2)

        # Convolutional path
        conv_out = None
        if self.conv_out is not None:
            conv_out = self.conv_out(x)

        # Combine paths
        if conv_out is not None:
            combined = torch.cat((conv_out, hyena_out), dim=1)
        else:
            combined = hyena_out

        return self.proj_out(combined)