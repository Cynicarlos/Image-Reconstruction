import argparse
import os
import parser
import random
import time
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from models.NAFNet.NAFNet import NAFNet
from models.UNet.UNet import UNet
from models.KBNet.KBNet import KBNet_s
from utils.custom_dataset import CustomDataset
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


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def create_model(type, local_rank):
    if type == 'UNet':
        if local_rank == 0:
            print('create UNet model')
        model = UNet(1,1)
    elif type == 'NAFNet':
        if local_rank == 0:
            print('create NAFNet model')
        model = NAFNet(3)
    elif type == 'KBNet':
        if local_rank == 0:
            print('create KBNet model')
        model = KBNet_s(img_channel=3)
    else:
        raise ValueError('Wrong model name!')
    model.eval()
    return model

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def validate(val_loader, model, criterion, local_rank, args):
    losses = AverageMeter()
    val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, (images, target, _ ) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            losses.update(reduced_loss.item(), images.size(0))
            val_loss += loss.item()
    return val_loss/len(val_loader)


def main_worker(local_rank, nprocs, args):
    min_val_loss = float('inf')
    dist.init_process_group(backend='nccl')
    model = create_model(args.model, local_rank)
    if os.path.isfile('model_weights/best_model.pth'):
        model.load_state_dict(torch.load('model_weights/best_model.pth'))
    args.batch_size = int(args.batch_size / nprocs)
    torch.cuda.set_device(local_rank)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    # Data loading code
    data_root = os.path.join(args.data)
    tsfm = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(data_root, 'train', tsfm)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,sampler=train_sampler)

    val_dataset = CustomDataset(data_root, 'val', tsfm)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,sampler=val_sampler)
    writer = SummaryWriter('logs')
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        losses = AverageMeter()
        train_loss = 0.0
        model.train()
        for i, (images, target, _) in enumerate(train_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            losses.update(reduced_loss.item(), images.size(0))
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if local_rank == 0 and i % args.print_freq == 0:
            #     print(f'epoch [{epoch+1}/{args.epochs}]   batch [{i}]  train_loss: {losses.avg:.5f} lr: {scheduler.get_last_lr()[0]:.5f}')

        train_loss = train_loss/len(train_loader)
        val_loss = validate(val_loader, model, criterion, local_rank, args)
        if local_rank == 0:
            print(f'epoch[{epoch+1}/{args.epochs}]  train_loss:{train_loss:.5f} val_loss:{val_loss:.5f} lr: {scheduler.get_last_lr()[0]:.5f}')
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                if local_rank == 0:
                    model_info ={
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'min_val_loss': min_val_loss,
                        }
                    torch.save(model_info, 'model_weights/best_model.pth')

            writer.add_scalars('Loss', {
                'train_loss' : train_loss,
                'val_loss' : val_loss 
            }, global_step=epoch)
        scheduler.step()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IAT Lab')
    parser.add_argument('--data',default='datasets/SenseNoise')
    parser.add_argument('--model',default='NAFNet',choices=['UNet', 'NAFNet','KBNet'])
    parser.add_argument('--workers',default=4,type=int)
    parser.add_argument('--epochs',default=100,type=int)
    parser.add_argument('--start_epoch',default=0,type=int)
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--local_rank',default=-1,type=int)
    parser.add_argument('--weight_decay',default=1e-4,type=float)
    parser.add_argument('--print-freq',default=50,type=int)
    parser.add_argument('--pretrained',dest='pretrained',action='store_true')
    parser.add_argument('--seed',default=42,type=int)
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_worker(args.local_rank, args.nprocs, args)

