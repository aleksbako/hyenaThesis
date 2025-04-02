
import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyena.Hyena_SEBlock import SE_Block_Hyena

class MobileNetWithHyena(nn.Module):
    def __init__(self, num_classes=100, input_channels=3, input_size=32, dropout=0.05):
        super(MobileNetWithHyena, self).__init__()
        self.input_size = input_size  # Store input size for dynamic adaptation

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SE_Block_Hyena(64,64)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SE_Block_Hyena(128,128)
        self.dropout2 = nn.Dropout(p=dropout)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)       
        self.se3 = SE_Block_Hyena(256,256)
        self.dropout3 = nn.Dropout(p=dropout)

        # Dynamically calculate the size of the fully connected layer
        self.fc = nn.Linear(256, num_classes)
        self.dropout_fc = nn.Dropout(p=dropout)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.dropout3(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)  # Dropout before the fully connected layer
        x = self.fc(x)
        return x