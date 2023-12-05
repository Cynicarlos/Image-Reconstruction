import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, dataset_type='train', transform=None):
        self.root = root
        self.transform = transform
        self.Images = []  # 存储Image文件路径
        self.GTs = []  # 存储GT文件路径

        # 获取图像和标签文件路径
        if dataset_type == 'train':
            self.Image_dir = os.path.join(root, 'train/Image')
            self.GT_dir = os.path.join(root, 'train/GT')
            self.Images = os.listdir(self.Image_dir)
            self.GTs = os.listdir(self.GT_dir)
        elif dataset_type == 'val':
            self.Image_dir = os.path.join(root, 'val/Image')
            self.GT_dir = os.path.join(root, 'val/GT')
            self.Images = os.listdir(self.Image_dir)
            self.GTs = os.listdir(self.GT_dir)
        elif dataset_type == 'test':
            self.Image_dir = os.path.join(root, 'test/Image')
            self.GT_dir = os.path.join(root, 'test/GT')
            self.Images = os.listdir(self.Image_dir)
            self.GTs = os.listdir(self.GT_dir)
        else:
            raise ValueError("Wrong dataset_type! dataset_type should be 'train', 'val' or 'test'")

    def __getitem__(self, idx):
        Image_path = os.path.join(self.Image_dir,self.Images[idx])
        GT_path = os.path.join(self.GT_dir,self.GTs[idx])
        image = Image.open(Image_path)
        GT = Image.open(GT_path)
        if self.transform:
            image = self.transform(image)
            GT = self.transform(GT)

        return image, GT, self.Images[idx]

    def __len__(self):
        return len(self.Images)
    
'''
if __name__ == "__main__": 
    data_transform = transforms.Compose([
        # transforms.CenterCrop((1024,1024)),
        transforms.ToTensor()
    ])
    root = './datasets/SIDD'
    train_dataset = CustomDataset(root=root, dataset_type='train', transform=data_transform)
    image, GT, _ = train_dataset[77]
    print(image.shape)
'''

