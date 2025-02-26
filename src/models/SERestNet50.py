import torch
import torch.nn as nn


class SEResNet50(nn.Module):
    def __init__(self, layerType,num_classes=1000, include_top=True, dropout_rate=0.1):
        super(SEResNet50, self).__init__()
        self.inplanes = 64
        self.include_top = include_top
        self.groups = 1
        self.base_width = 64

        # Stem layer (initial convolution and pooling)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        """
        # ResNet layers (with SE Bottleneck blocks)
        self.layer1 = self._make_layer(layerType, 64, 2,stride=1)
        self.layer2 = self._make_layer(layerType, 128, 2, stride=2)
        self.layer3 = self._make_layer(layerType, 256, 2, stride=2)
        self.layer4 = self._make_layer(layerType, 512, 2, stride=2)
        """
        #check out the following:
        
        self.layer1 = self._make_layer(layerType, 64, 1,stride=1)
        self.layer2 = self._make_layer(layerType, 128, 1, stride=2)
        self.layer3 = self._make_layer(layerType, 256, 1, stride=2)
        self.layer4 = self._make_layer(layerType, 512, 1, stride=2)
        
        # Final layers (average pooling and fully connected)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * layerType.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create a ResNet layer with SEBottleneck blocks."""
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution and max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Apply ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Dropout applied here
        x = self.fc(x)

        return x
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)