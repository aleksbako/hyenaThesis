
import torch.nn as nn
import torch
from ..Semantic.util.SqueezeAttentionBlock import SqueezeAttentionBlock
from ..Semantic.util.DialatedFCN import DilatedConvBlock
class SANetSimple(nn.Module):
      #Follows the example drawing in the paper
      def __init__(self,num_classes):
         super(SANetSimple,self).__init__()
         ch_in_stage1 = 3  # Input channels for stage 1 (e.g., RGB)
         ch_out_stage1 = 64  # Output channels for stage 1

         ch_in_stage2 = 64  # Input channels for stage 2 (output from stage 1)
         ch_out_stage2 = 128  # Output channels for stage 2

         ch_in_stage3 = 128  # Input channels for stage 3 (output from stage 2)
         ch_out_stage3 = 256  # Output channels for stage 3

         ch_in_stage4 = 256  # Input channels for stage 4 (output from stage 3)
         ch_out_stage4 = 512  # Output channels for stage 4

         self.stage1 = DilatedConvBlock(ch_in_stage1,ch_out_stage1)
         self.SA1 = SqueezeAttentionBlock(ch_out_stage1,ch_out_stage2)
         self.stage2 = DilatedConvBlock(ch_in_stage2,ch_out_stage2)
         self.SA2 = SqueezeAttentionBlock(ch_out_stage2,ch_in_stage3)
         self.stage3 = DilatedConvBlock(ch_in_stage3,ch_out_stage3)
         self.SA3 = SqueezeAttentionBlock(ch_out_stage3,ch_in_stage4)
         self.stage4 = DilatedConvBlock(ch_in_stage4,ch_out_stage4)
         self.SA4 = SqueezeAttentionBlock(ch_out_stage4,ch_out_stage4*2)
           # Add an average pooling layer
         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

         # Add an additional FC layer for categorical_predicted_y
         self.fc_y = nn.Linear(sum(ch_out_stage2, ch_out_stage3, ch_out_stage4, ch_out_stage4 * 2), num_classes)
         self.fc = nn.Linear(sum(ch_out_stage2,ch_out_stage3,ch_out_stage4,ch_out_stage4*2), num_classes)
         self.fcn_head = nn.Conv2d(ch_out_stage4 * 2, num_classes, kernel_size=1)

      def forward(self,x):
         x = self.stage1(x)
         sa_1_x = self.SA1(x)
         x = self.stage2(x)
         sa_2_x = self.SA2(x)
         x = self.stage3(x)
         sa_3_x = self.SA3(x)
         x = self.stage4(x)
         sa_4_x = self.SA4(x)
         #FCN for pixel-wise

         #FC Layer
         sa_aggregated = torch.cat([sa_1_x,sa_2_x,sa_3_x,sa_4_x], dim=1)  # Concatenate along the channel dimension
         class_wise_mask = self.fc(sa_aggregated)

               # Calculate categorical_predicted_y
         pooled = self.avg_pool(sa_aggregated)
         pooled = pooled.view(pooled.size(0), -1)  # Flatten the tensor
         categorical_pred_y = self.fc_y(pooled)

         y = self.fcn_head(x)
         y = nn.functional.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
         return class_wise_mask, categorical_pred_y, y