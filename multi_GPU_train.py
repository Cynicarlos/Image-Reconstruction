import argparse
import os
import parser
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from models.NAFNet.NAFNet import NAFNet
from models.UNet.UNet import UNet
from utils.custom_dataset import CustomDataset
from utils.loss import WeightedL1Loss
from utils.Helper import Helper
import torch.distributed as dist
from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  # 总进程数
    return rt

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

def check_best_model(path):
    if os.path.isfile(path):
        return True
    else:
        return False

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, optimizer, validation, args):
    writer = SummaryWriter('logs')
    best_model_path = os.path.join(*[args.model_dir, 'best_model.pth'])
    start_epoch = 0
    min_val_loss = 9999
    if check_best_model(best_model_path):
        best_state = torch.load(best_model_path)
        model.load_state_dict(best_state['model'])
        min_val_loss = best_state['valid_loss']
        print('Using best model!')
        if args.resume:
            if args.last_finished_epoch > 0:
                start_epoch = args.last_finished_epoch
            else:
                start_epoch = best_state['epoch']
        print(f'Start training model from epoch {start_epoch}!')
    else:
        print('Start training model from scratch!')
    valid_losses = []
    scheduler = CosineAnnealingLR(optimizer, args.n_epoch, eta_min=7e-5)


    for epoch in range(start_epoch, start_epoch + args.n_epoch):
        losses = AverageMeter()
        model.train()
        for batch_idx, (Image, GT, _) in enumerate(train_loader):
            two_pro_loss = 0
            Image, GT = Image.cuda(args.local_rank, non_blocking=True), GT.cuda(args.local_rank, non_blocking=True)
            output = model(Image)
            current_weights = Helper.GetLossWeight(GT)
            weighted_l1_loss = WeightedL1Loss(weight=current_weights)
            loss = weighted_l1_loss(output, GT)
            two_pro_loss += reduce_tensor(loss).item() # 有多个进程，把进程0和1的loss加起来平均
            # two_pro_loss = two_pro_loss.cuda(args.local_rank, non_blocking=True)
            losses.update(two_pro_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = losses.avg
        valid_loss = validation(model, valid_loader)
        valid_losses.append(valid_loss)
 
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        model_info = {
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': train_loss
        }
        if epoch % 20 == 0:
            torch.save(model_info, epoch_model_path)
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            torch.save(model_info, best_model_path)
        if torch.cuda.current_device() == 0:
            Helper.RecordLossInfo(epoch, args.n_epoch + start_epoch, train_loss, valid_loss, optimizer)
            writer.add_scalars('Loss',{
                'Train Loss' : train_loss,
                'Valid Loss' : valid_loss,
            }, global_step=epoch)
        scheduler.step()
    writer.close()

def validate(model, val_loader):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, (Image, GT, _) in enumerate(val_loader):
            two_pro_loss = 0
            Image = Image.cuda(args.local_rank, non_blocking=True)
            GT = GT.cuda(args.local_rank, non_blocking=True)
            output = model(Image)
            current_weights = Helper.GetLossWeight(GT)
            weighted_l1_loss = WeightedL1Loss(weight=current_weights)
            loss = weighted_l1_loss(output, GT)
            two_pro_loss += reduce_tensor(loss).item()
            losses.update(two_pro_loss)
    return losses.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch IAT Training')
    parser.add_argument('--n_epoch', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-2, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', default=8, type=int, help='num_workers')
    parser.add_argument('--data_dir',type=str, default='datasets/IAT', help='input dataset directory')
    parser.add_argument('--model_dir', type=str, default='model_weights', help='output model directory')
    parser.add_argument('--model_type', type=str, required=False, default='UNet', choices=['UNet', 'NAFNet'])
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--last_finished_epoch',type=int, default=-1)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    # os.environ['LOCAL_RANK'] = -1

    dist.init_process_group(backend='nccl')

    os.makedirs(args.model_dir, exist_ok=True)

    model = create_model(args.model_type)
    optimizer  =torch.optim.Adam(model.parameters(),args.lr)

    tfms = transforms.Compose([transforms.CenterCrop((896,896)),
                               transforms.ToTensor()])
    train_dataset = CustomDataset(root=args.data_dir,dataset_type='train',transform=tfms)
    val_dataset = CustomDataset(root=args.data_dir,dataset_type='val',transform=tfms)

    train_sample = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sample = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sample)
    valid_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=valid_sample)

    model.cuda(args.local_rank)

    train(train_loader, model, optimizer, validate, args)

