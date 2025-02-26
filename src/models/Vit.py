import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Vit(nn.Module):
    def __init__(self, classNumber=257, preTrained=False):
        super(Vit, self).__init__()
        self.ViT = torchvision.models.vit_b_16(pretrained=preTrained, dropout=0.4).to("cuda")
        
        # Reset the weights of MultiheadAttention layers
        self.named_modules_copy = dict(self.ViT.named_modules())
        for name, module in self.named_modules_copy.items():
            if isinstance(module, nn.MultiheadAttention):
                pass
                # Reset the weights of the MultiheadAttention layer
              #  module._reset_parameters()

        self.ViT.heads.head = nn.Linear(768, classNumber)  # Adjust the input size to match the output size of the ViT model
          # Remove the existing linear layer
       

    def forward(self, x):
        return self.ViT(x)
