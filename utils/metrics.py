import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR

def calc_ssim(predicted, target):
    predicted = torch.clamp(predicted, min=0, max=1)
    predicted = predicted[0][0]
    target = target[0][0]
    predicted, target = predicted.cpu().numpy(), target.cpu().numpy()
    # 此处如果图像范围是0-255，data_range为255，如果是为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = SSIM(predicted, target, data_range=1)
    return ssim_score

def calc_PSNR(predicted, target):   
    predicted = torch.clamp(predicted, min=0, max=1)
    predicted = predicted[0][0]
    target = target[0][0]
    predicted, target = predicted.cpu().numpy(), target.cpu().numpy()
    psnr = PSNR(target, predicted)
    return psnr

def dice_coefficient(predicted, target):
    """
    计算 Dice 系数
    :param predicted: 模型预测的二值分割掩码
    :param target: 真实的二值分割掩码
    :return: Dice 系数
    """
    intersection = np.logical_and(predicted, target)
    return 2.0 * intersection.sum() / (predicted.sum() + target.sum())


def calculate_iou(predicted, target):
    pass


def calculate_accuracy(predicted, target):
    predicted = torch.clamp(predicted, min=0, max=1)
    predicted = (255*predicted).round()
    target = 255*target

    # 计算正确预测的样本数
    correct = (predicted == target).sum().item()
    # 计算准确率
    total = len(target[0][0])**2
    accuracy = correct / total
    return accuracy
