import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from models.NAFNet import NAFNet  # 导入你的模型定义
from PIL import Image

def predict(model, image_path):
    model.eval()
    transform = transforms.Compose([
        # transforms.Resize((1280,1280)),
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
    model = NAFNet(img_channel=1)
    model.to('cuda')
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 图像路径
    image_dir = 'E:/data/Image_croped'
    label_dir = 'E:/data/GT_croped'

    
    #read txt method three
    f = open("./datasets/IAT/partition/test.txt","r")
    lines = f.read().splitlines()
    image_label_list = []
    # 生成多个随机三位数并添加到列表中
    for _ in range(2):
        random_test_name = lines[random.randint(0, len(lines)-1)]
        image_path = os.path.join(image_dir,str(random_test_name))
        label_path = os.path.join(label_dir,str(random_test_name))
        image_label_list.append([image_path,label_path])

    plt.figure(figsize=(8, 8))
    for i in range(len(image_label_list)):
        image_path = image_label_list[i][0]
        label_path = image_label_list[i][1]
        image = Image.open(image_path)
        label = Image.open(label_path)
        # 进行预测 
        prediction = predict(model, image_path)
        # 截断
        # prediction = torch.clamp(prediction, min=0, max=1)
        # prediction = 255*prediction
        prediction = prediction[0,0].cpu().detach().numpy()
        plt.subplot(len(image_label_list), 2, 2*i+1)
        plt.imshow(label,cmap='gray')
        if i == 0:
            plt.title('GT')

        plt.subplot(len(image_label_list), 2, 2*i+2)
        plt.imshow(prediction, cmap='gray')
        if i == 0:
            plt.title('Prediction')
    plt.show()
