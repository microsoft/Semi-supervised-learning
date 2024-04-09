import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

from ..dataset_utils import *

from torch.utils.data import  Dataset
import numpy as np
from PIL import Image


class Cifar10Data(Dataset):
    def __init__(self,data_conf):
        self.data_conf = data_conf
        self.d = self.data_conf["dimension"]
        self.num_classes = 10 
        self.f = None
        
    def __getitem__(self, index):
        
        x = self.X[index]
        
        #if self.transform:
        #    x = Image.fromarray(x.numpy().astype(np.uint8))
        #    x = self.transform(x)     
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
            #print(len(self.Y))
            Y = self.Y[idcs] 
            return CustomTensorDataset(X=X,Y=Y,num_classes = self.num_classes, d=self.d,transform=self.transform)

    def build_dataset(self):
        data_dir  = self.data_conf['data_path']
        
        self.transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        #[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        #self.transform = transform_train = transforms.Compose([
        #    transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #]) for resnet

        self.data = torchvision.datasets.CIFAR10(root=self.data_conf['data_path'], 
                                                 train=True, 
                                                 download=True, 
                                                 transform=self.transform)
        
        self.test_data  = torchvision.datasets.CIFAR10(root=self.data_conf['data_path'], 
                                                       train=False,
                                                       download=True, 
                                                       transform=self.transform)
        
        self.X =  torch.tensor(self.data.data).float()
        #print(self.X.shape)

        self.X =  self.X.permute(0,3,1,2)
    
        
        self.Y = torch.tensor(self.data.targets) 

        self.X_test =  torch.tensor(self.test_data.data).float()
        # print(self.X_test.shape)
        
        self.X_test = self.X_test.permute(0,3,1,2)

        self.Y_test =  torch.tensor(self.test_data.targets).float()
        
        if "flatten" in self.data_conf.keys() and self.data_conf['flatten'] == True: # flat the image from 28x28 to 784
            self.X =  self.X.reshape(self.X.shape[0],-1).float()
            self.X_test =  self.test_data.data.reshape(self.test_data.data.shape[0],-1).float()
            self.transform = None
        
    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return CustomTensorDataset(X=X_,Y=Y_, num_classes = self.num_classes,d=self.d,transform=self.transform)
