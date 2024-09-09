from Hyena.HyenaOperator import HyenaOperator
import torchvision
import torch.nn as nn
import torch.nn.functional as F
class HyenaVit(nn.Module):
    def __init__(self, classNumber=257, preTrained=False):
        super(HyenaVit, self).__init__()
        self.ViT = torchvision.models.vit_b_16(pretrained=True).to("cuda")
        # Replace the attention layers with HyenaOperator
        self.named_modules_copy = dict(self.ViT.named_modules())
        for name, module in self.named_modules_copy.items():
            if isinstance(module, nn.MultiheadAttention):
                setattr(self.ViT, name, HyenaOperator(
                    d_model=module.embed_dim,
                    l_max=196,  # You can adjust this value according to your requirements
                    order=2,  # You can adjust this value according to your requirements
                    filter_order=64, 
                     dropout=0.1,
                      filter_dropout=0.1 # You can adjust this value according to your requirements
                ))
        self.ViT.head = nn.Identity()  # remove the existing linear layer
        self.new_head = nn.Sequential(
            nn.Linear(1000, classNumber),  # Adjust the input size to match the output size of the ViT model
        )

    def get_final_attention_layer(self):
        # Access the final HyenaOperator layer
        for name, module in reversed(list(self.ViT.named_modules())):
            if isinstance(module, HyenaOperator):
                return module

        

    def forward(self, x):
        x = self.ViT(x)
        return self.new_head(x)