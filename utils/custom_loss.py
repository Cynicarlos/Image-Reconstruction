import torch
import torch.nn as nn

class Custom_L1_Loss(nn.Module):
    def __init__(self, weight_background, weight_target):
        super(Custom_L1_Loss, self).__init__()
        self.weight_background = weight_background
        self.weight_target = weight_target

    def forward(self, output, target):
        # 定义你的损失计算逻辑
        # 例如，可以使用L1损失，并在计算中应用权重
        loss_target = nn.L1Loss()(output * target, target) * self.weight_target
        loss_background = nn.L1Loss()(output * (1 - target), torch.zeros_like(output)) * self.weight_background
        total_loss = loss_target + loss_background
        return total_loss
