import torch
import torch.nn as nn

class DilatedConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
