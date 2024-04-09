import torchvision
from torchvision import transforms
import numpy as np
import torch
from ..dataset_utils import *
from torch.utils.data import  Dataset
import numpy as np


class STL10Data(Dataset):
    def __init__(self,data_conf):
        self.data_conf = data_conf
        self.d = self.data_conf["dimension"] # should be 28x28
        self.num_classes = 10 
        self.f = None
        
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y, index
    
    def __len__(self):
        return len(self.X)
    
    def len(self):
        return len(self.X)
    
    def get_subset(self,idcs):
        idcs = np.array(idcs) 
        X = None
        Y = None 
        if(self.X is not None):
            X = self.X[idcs] 
        if(self.Y is not None):
            print(len(self.Y))
            Y = self.Y[idcs] 
            return CustomTensorDataset(X=X,Y=Y,num_classes = self.num_classes, d=self.d,transform=self.transform)

    def build_dataset(self):
        # Why I use 0.5 here
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))
            ]
        )

        self.data = torchvision.datasets.STL10(
            root=self.data_conf['data_path'],
            split="train", download=True, transform=self.transform)
        self.test_data = torchvision.datasets.STL10(
            root=self.data_conf['data_path'],
            split="test", download=True, transform=self.transform)

        self.X = torch.tensor(self.data.data).float()

        self.Y = torch.tensor(self.data.labels) 

        self.X_test = torch.tensor(self.test_data.data).float()

        self.Y_test = torch.tensor(self.test_data.labels).float()
        
        if "flatten" in self.data_conf.keys() and self.data_conf['flatten'] == True:
            self.X = self.X.reshape(self.X.shape[0], -1).float()
            self.X_test = torch.tensor(self.test_data.data.reshape(self.test_data.data.shape[0],-1)).float()
            self.transform = None
        
    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return CustomTensorDataset(X=X_,Y=Y_, num_classes = self.num_classes,d=self.d,transform=self.transform)
