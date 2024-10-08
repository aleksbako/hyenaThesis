import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ModifiedVit(nn.Module):
    def __init__(self, pretrained_model, classNumber=257):
        super(ModifiedVit, self).__init__()
        self.pretrained = pretrained_model
        self.pretrained.head = nn.Identity()  # remove the existing linear layer
        self.new_head = nn.Sequential(
            nn.Linear(1000, classNumber),
            nn.ReLU()
            # Adjust the input size to match the output size of the ViT model
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.new_head(x)
        x = F.relu(x)
        return x