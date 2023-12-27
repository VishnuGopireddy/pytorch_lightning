import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class ImageDataset(LightningDataModule):
    def __init__(self, datadir, img_size, batchsize,
                 num_workers, split_train_val_test=True):
        super().__init__()
        self.datadir = datadir
        self.batchsize = batchsize
        self.img_size = img_size
        self.num_workers = num_workers
        self.split_train_val_test = split_train_val_test
        self.prepare_data_per_node=True
        
    def setup(self, stage=None):

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.RandomPerspective(p=0.5),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
            ])
        
        test_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])        
        
        entire_dataset = ImageFolder(self.datadir)
        if self.split_train_val_test:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(entire_dataset,
                                                                                   [0.7, 0.15, 0.15],
                                                                                   generator=torch.Generator().manual_seed(42))
            self.train_dataset.dataset.transform = train_transform
            self.val_dataset.dataset.transform = test_transform
            self.test_dataset.dataset.transform = test_transform
            self.train_dataset.dataset.shuffle = True
            self.val_dataset.dataset.shuffle = True
        else:
            train_path = os.path.join(self.datadir, 'train')
            val_path = os.path.join(self.datadir, 'val')
            test_path = os.path.join(self.datadir, 'test')
            self.train_dataset = ImageFolder(train_path, transform=train_transform, shuffle=True)
            self.val_dataset = ImageFolder(val_path, transform=test_transform, shuffle=True)
            self.test_dataset = ImageFolder(test_path, transform=test_transform, shuffle=False)
        return self.train_dataset, self.val_dataset, self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batchsize, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batchsize, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batchsize, num_workers=self.num_workers)