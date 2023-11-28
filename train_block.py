import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.loss import WeightedL1Loss
from utils.Helper import Helper
from utils.block_process import block_process

def main():
    resume, num_epochs, batch_size, device, initial_lr, model_name,data_root = Helper.DefineHyperparameter()
    #加载模型
    model = Helper.CreateModel(model_name=model_name)
    model.to(device)

    #准备数据集
    data_transforms = transforms.Compose([
        # transforms.CenterCrop((1024,1024)),
        transforms.ToTensor()
    ])
    train_loader, val_loader = Helper.get_Train_Val_Dataloader(data_root, batch_size=batch_size, transforms=data_transforms, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    writer = SummaryWriter('logs')  # 指定日志文件夹的路径
    min_val_loss = float('inf')
    last_epoch = 0
    if resume:
        with open('last_epoch.txt','r') as f:
            last_epoch = int(f.read())
    print('Start Training\n--------------------------------------------------------------------------------------------------------')
    total_epoch = last_epoch + num_epochs
    for epoch in range(last_epoch, total_epoch):
        min_val_loss = train_val_one_epoch(model, device, optimizer, train_loader, val_loader, epoch, total_epoch, scheduler, writer, min_val_loss)
        with open('last_epoch.txt','w') as f:
            f.write(str(epoch))
        
    writer.close()

def train_val_one_epoch(model, device, optimizer, train_loader, val_loader, epoch, total_epoch, scheduler, writer, min_val_loss):
    model.train()
    total_train_loss = 0.0
    for batch_idx, (Image, GT, _) in enumerate(train_loader):
        Image, GT = Image.to(device), GT.to(device)
        optimizer.zero_grad()
        output = block_process(Image, model, (384,384),(64,64))
        output.requires_grad_(True)
        current_weights = Helper.GetLossWeight(output, GT)
        weighted_l1_loss = WeightedL1Loss(weight=current_weights)
        loss = weighted_l1_loss(output, GT)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    average_train_loss = total_train_loss / len(train_loader)

    # 在验证集上评估模型
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (Image, GT, _) in enumerate(val_loader):
            Image, GT = Image.to(device), GT.to(device)
            output = block_process(Image, model, (384,384),(64,64))
            output.requires_grad_(True)
            current_weights = Helper.GetLossWeight(output, GT)
            weighted_l1_loss = WeightedL1Loss(weight=current_weights)
            loss = weighted_l1_loss(output, GT)
            total_val_loss += loss.item()
    average_val_loss = total_val_loss / len(val_loader)

    writer.add_scalars('Loss', tag_scalar_dict={
        'Train Loss' : average_train_loss,
        'Validation Loss' : average_val_loss
    }, global_step=epoch+1)

    Helper.RecordLossInfo(epoch, total_epoch, average_train_loss, average_val_loss, optimizer)
    
    #更新学习率
    scheduler.step()

    #每20个epoch保存一次模型
    if((epoch+1) % 20 == 0):
        torch.save(model.state_dict(),'model_weights/{}.pth'.format(epoch+1))

    #保存最佳模型
    if average_val_loss < min_val_loss:
        min_val_loss = average_val_loss
        torch.save(model.state_dict(), "model_weights/best_model.pth")
    return min_val_loss


if __name__ == '__main__':
    main()

