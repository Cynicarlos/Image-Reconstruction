from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from models.NAFNet.NAFNet import NAFNet  # 导入你的模型定义
from models.UNet.UNet import UNet
from utils.custom_dataset import CustomDataset  # 导入数据加载器
from utils.metrics import calc_ssim, calc_PSNR # 导入性能指标计算函数
from utils.Helper import Helper

def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterionL1 = nn.L1Loss()
    criterionL2 = nn.MSELoss()
    total_L1_loss = 0.0
    total_L2_loss = 0.0
    total_SSIM = 0.0
    total_PSNR = 0.0
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            L1_loss = criterionL1(outputs, labels)
            total_L1_loss += L1_loss.item()
            
            L2_loss = criterionL2(outputs, labels)
            total_L2_loss += L2_loss.item()

            SSIM = calc_ssim(outputs, labels)
            total_SSIM += SSIM

            PSNR = calc_PSNR(outputs, labels)
            total_PSNR += PSNR
            
    average_L1_loss = total_L1_loss / len(dataloader)
    average_L2_loss = total_L2_loss / len(dataloader)
    average_SSIM = total_SSIM/len(dataloader)
    average_PSNR = total_PSNR/len(dataloader)


    
    return average_L1_loss, average_L2_loss, average_SSIM, average_PSNR

if __name__ == "__main__":
    # 加载模型
    model = UNet(1,1)
    checkpoint = torch.load('./model_weights/best_model.pth')
    model.load_state_dict(checkpoint['model'])

    data_transforms = transforms.Compose([
        transforms.CenterCrop((640,640)),
        transforms.ToTensor()
    ])
    helper = Helper()
    test_loader = helper.get_Test_Dataloader('./datasets/IAT',batch_size=2, transforms=data_transforms,shuffle=True)
    train_loader, _ = helper.get_Train_Val_Dataloader('./datasets/IAT',batch_size=2, transforms=data_transforms,shuffle=True)
    L1_loss, L2_loss, SSIM, PSNR = evaluate_model(model, test_loader)
    with open('evaluate.txt', "w") as f:
        f.write(f'L1_loss(MAE) : {L1_loss}, L2_Loss(MSE) : {L2_loss}, SSIM : {SSIM}, PSNR : {PSNR}')
    print(f'L1_loss(MAE) : {L1_loss}, L2_Loss(MSE) : {L2_loss}, SSIM : {SSIM}, PSNR : {PSNR}')
