import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from models.NAFNet.NAFNet import NAFNet  # 导入你的模型定义
from models.UNet.UNet import UNet
from PIL import Image
from torchvision.utils import save_image
def predict(model, image_path):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.to('cuda').unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
    
    return output

if __name__ == "__main__":
    # 加载模型
    model = UNet(1, 1)
    model.to('cuda')
    checkpoint = torch.load('./model_weights/best_model.pth')
    model.load_state_dict(checkpoint['model'])
    
    # 图像路径
    image_dir = './datasets/IAT/test/Image'
    save_path = 'predict_results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Tensor转PIL
    Tensor2PIL = transforms.ToPILImage()
    f = open("./datasets/IAT/partition/test.txt","r")
    lines = f.read().splitlines()
    for i in range(len(lines)):
        image_path = os.path.join(image_dir,lines[i])
        image = Image.open(image_path)
        prediction = Tensor2PIL(predict(model, image_path)[0])
        output_path = os.path.join(save_path, lines[i])
        prediction.save(output_path)
