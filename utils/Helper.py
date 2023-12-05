import os
import torch
from torch.utils.data import DataLoader
import yaml
from torch.nn.parallel import DistributedDataParallel
from models.NAFNet.NAFNet import NAFNet
from models.UNet.UNet import UNet
from utils.custom_dataset import CustomDataset
from torch.utils.data.distributed import DistributedSampler


class Helper:
    @staticmethod
    def CreateModel(model_name):
        if model_name == 'UNet':
            model = UNet(in_channels=1, out_channels=1)
        elif model_name == 'NAFNet':
            model = NAFNet(img_channel=1)
        else:
            raise ValueError('Wrong model_name, please check again')
        if os.path.isfile('./model_weights/best_model.pth'):
            model.load_state_dict(torch.load('./model_weights/best_model.pth'))
            print('Using best model')
        else:
            print('training from scratch!')
        return model
    
    @staticmethod
    def get_Train_Val_Dataloader(data_root,batch_size, transforms=None, shuffle=True):
        train_data = CustomDataset(data_root,'train',transforms)
        val_data = CustomDataset(data_root,'val',transforms)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_loader
    
    @staticmethod
    def get_Train_Val_Dataloader_With_Distribution(data_root,batch_size,world_size,rank,transforms=None):
        train_data = CustomDataset(data_root,'train',transforms)
        val_data = CustomDataset(data_root,'val',transforms)
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_data, batch_size=batch_size//world_size,sampler=train_sampler)
        val_loader = DataLoader(val_data, batch_size=batch_size//world_size,sampler=val_sampler)
        return train_loader, val_loader
    
    @staticmethod
    def get_Test_Dataloader(data_root, batch_size, transforms=None, shuffle=True):
        test_data = CustomDataset(data_root,'test',transforms)
        return DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def DefineHyperparameter():
        with open('./config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        resume = config['resume']
        num_epochs = config['num_epochs']
        batch_size = config['batchsize']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        initial_lr = float(config['initial_lr'])
        model_name = config['model']
        data_root = config['dataset']['root']
        return resume, num_epochs, batch_size, device, initial_lr, model_name, data_root
    
    @staticmethod
    def DefineLossParameter():
        with open('./config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        white_lower = config['weighted_loss']['white_lower']  # 白色目标的下限
        white_upper = config['weighted_loss']['white_upper']  # 白色目标的上限
        white_weight = config['weighted_loss']['white_weight']
        black_weight = config['weighted_loss']['black_weight']

        return white_lower, white_upper, white_weight, black_weight
    
    @staticmethod
    def GetLossWeight(GT):
        """
        Parameters:
        - GT(Tensor)

        Returns:
        tensor: The weighted tensor on different GT.
        """
        current_weights = torch.zeros_like(GT)
        white_lower, white_upper, white_weight, black_weight = Helper.DefineLossParameter()
        current_weights[(GT >= white_lower) & (GT <= white_upper)] = white_weight
        current_weights[(GT < white_lower)] = black_weight
        return current_weights
    
    @staticmethod
    def RecordLossInfo(epoch, num_epochs, average_train_loss, average_val_loss, optimizer,loss_info_path='./loss.txt'):
        with open(loss_info_path, "a") as file:
            file.write(f"Epoch {epoch+1}/{num_epochs},Train Loss: {average_train_loss:.6f}, Valid Loss: {average_val_loss:.6f}, lr:{float(optimizer.param_groups[0]['lr']):.6f}" +"\n")
        # print(f"Epoch {epoch+1}/{num_epochs},Train Loss: {average_train_loss:.6f}, Valid Loss: {average_val_loss:.6f}, lr:{float(optimizer.param_groups[0]['lr']):.6f}")
