import torch.nn as nn
import torch.nn.functional as F
from .Hyena_AugmentedConv import Hyena_AugmentedConv
class Hyena_WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, shape, stride=1, order=2):
        super(Hyena_WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv1 = Hyena_AugmentedConv(
            in_channels=in_planes,
            out_channels=planes,
            order=order,
            dropout=dropout_rate,
            stride=stride,
            filter_dropout=0.0
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Hyena_AugmentedConv(
            in_channels=planes,
            out_channels=planes,
            order=order,
            dropout=dropout_rate,
            stride=stride,
            filter_dropout=0.0
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                Hyena_AugmentedConv(
                    in_channels=in_planes,
                    out_channels=planes,
                    order=order,
                    stride=stride,
                    filter_dropout=0.1
                )
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        shortcut = self.shortcut(x)

        out += shortcut
        return out