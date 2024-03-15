from Hyena.HyenaOperator import HyenaOperator
import torchvision
import torch.nn as nn
import torch.nn.functional as F
class HyenaVit(nn.Module):
    def __init__(self, classNumber=257):
        super(HyenaVit, self).__init__()
        self.ViT = torchvision.models.vit_b_16(pretrained=False).to("cuda")
        # Replace the attention layers with HyenaOperator
        named_modules_copy = dict(self.ViT.named_modules())
        for name, module in named_modules_copy.items():
            if isinstance(module, nn.MultiheadAttention):
                setattr(self.ViT, name, HyenaOperator(
                    d_model=module.embed_dim,
                    l_max=196,  # You can adjust this value according to your requirements
                    order=2,  # You can adjust this value according to your requirements
                    filter_order=64,  # You can adjust this value according to your requirements
                ))
        self.ViT.head = nn.Identity()  # remove the existing linear layer
        self.new_head = nn.Sequential(
            nn.Linear(1000, classNumber),  # Adjust the input size to match the output size of the ViT model
        )

    def forward(self, x):
        x = self.ViT(x)
        x = self.new_head(x)
        x = F.relu(x)
        return x