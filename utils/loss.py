import torch
import torch.nn as nn

class WeightedL1Loss(nn.Module):
    def __init__(self, weight):
        super(WeightedL1Loss, self).__init__()
        self.weight = weight

    def forward(self, Image, GT):
        # 计算 L1 Loss
        l1_loss = torch.abs(Image - GT)

        # 对每个像素应用权重
        weighted_loss = l1_loss * self.weight

        # 返回加权 L1 Loss
        return weighted_loss.mean()