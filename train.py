import argparse
import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.NAFNet.NAFNet import NAFNet
from models.UNet.UNet import UNet
from utils.loss import WeightedL1Loss
from utils.Helper import Helper

def check_best_model(path):
    if os.path.isfile(path):
        return True
    else:
        return False

def create_model(type ='UNet'):
    if type == 'UNet':
        print('create UNet model')
        model = UNet(1,1)
    elif type == 'NAFNet':
        print('create NAFNet model')
        model = NAFNet(1)
    else:
        assert False
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch IAT Training')
    parser.add_argument('--num_epoch', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-2, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--data_dir',type=str, default='datasets/IAT', help='input dataset directory')
    parser.add_argument('--model_dir', type=str, default='model_weights', help='output model directory')
    parser.add_argument('--model_type', type=str, required=False, default='UNet', choices=['UNet', 'NAFNet'])
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--last_finished_epoch',type=int, default=100)
    parser.add_argument('--device',type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = args.device
    model = create_model(args.model_type).to(device)
    optimizer  =torch.optim.Adam(model.parameters(),args.lr)
    tfms = transforms.Compose([transforms.CenterCrop((640,640)),
                               transforms.ToTensor()])
    train_loader, val_loader = Helper.get_Train_Val_Dataloader(args.data_dir, batch_size=args.batch_size, transforms=tfms, shuffle=True)
    
    print('Start Training\n--------------------------------------------------------------------------------------------------------')
    train(train_loader, val_loader, device, model, optimizer, args)



def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (Image, GT, _) in enumerate(val_loader):
            Image, GT = Image.to(device), GT.to(device)
            output = model(Image)
            current_weights = Helper.GetLossWeight(GT)
            weighted_l1_loss = WeightedL1Loss(weight=current_weights)
            loss = weighted_l1_loss(output, GT)
            total_val_loss += loss.item()
    average_val_loss = total_val_loss / len(val_loader)
    return average_val_loss

def train(train_loader, val_loader, device, model, optimizer, args):
    writer = SummaryWriter('logs')
    start_epoch = 0
    min_val_loss = float('inf')
    best_model_path = os.path.join(*[args.model_dir, 'best_model.pth'])
    if check_best_model(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        # model.load_state_dict(checkpoint['model'])
        print('Using best model!')
        if args.resume:
            if args.last_finished_epoch > 0:
                start_epoch = args.last_finished_epoch
        print(f'Start training model from epoch {start_epoch}!')
    else:
        print('Start training model from scratch!')
    scheduler = CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)
    for epoch in range(start_epoch, start_epoch + args.num_epoch):
        model.train()
        total_train_loss = 0.0
        for batch_idx, (Image, GT, _) in enumerate(train_loader):
            Image, GT = Image.to(device), GT.to(device)
            optimizer.zero_grad()
            output = model(Image)
            current_weights = Helper.GetLossWeight(GT)
            weighted_l1_loss = WeightedL1Loss(weight=current_weights)
            loss = weighted_l1_loss(output, GT)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)

        valid_loss = validate(model, val_loader, device)
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            model_info = {
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': train_loss
        }
            torch.save(model_info, best_model_path)
        Helper.RecordLossInfo(epoch, args.num_epoch + start_epoch, train_loss, valid_loss, optimizer)
        writer.add_scalars('Loss',{
            'Train Loss' : train_loss,
            'Validation Loss' : valid_loss,
        }, global_step=epoch)
        scheduler.step()
    writer.close()

if __name__ == '__main__':
    main()

