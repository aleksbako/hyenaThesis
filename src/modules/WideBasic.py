import torch.nn as nn
import torch.nn.functional as F
from .AugmentedConv import AugmentedConv
class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, shape, stride=1, v=0.125, k=0.125, Nh=2):
        super(WideBasic, self).__init__()
        if stride == 2:
            original_shape = shape * 2
        else:
            original_shape = shape
        
        self.bn1 = nn.BatchNorm2d(in_planes)
     
        self.conv1 = AugmentedConv(in_planes, planes, kernel_size=3, dk=int(k * planes), dv=int(v * planes), Nh=Nh, relative=False, shape=original_shape)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = AugmentedConv(planes, planes, kernel_size=3, dk=int(k * planes), dv=int(v * planes), Nh=Nh, stride=stride, relative=False, shape=shape)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                AugmentedConv(in_planes, planes, kernel_size=3, dk=int(k * planes), dv=int(v * planes), Nh=Nh, relative=False, stride=stride, shape=shape),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        short = self.shortcut(x)
        out += short

        return out