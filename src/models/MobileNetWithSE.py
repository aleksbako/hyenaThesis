
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.SEBlock import SEBlock
class MobileNetWithSE(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetWithSE, self).__init__()
        # Define MobileNet layers (simplified for demonstration)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)  # SE Block after first convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)  # SE Block after second convolution
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)  # Apply SE Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)  # Apply SE Block
        x = F.avg_pool2d(x, x.size()[2:])  # Global average pooling
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x