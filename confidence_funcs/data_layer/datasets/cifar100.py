import torchvision
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

from ..dataset_utils import *

from torch.utils.data import  Dataset
import numpy as np
from PIL import Image


class Cifar100Data(Dataset):
    def __init__(self,data_conf):
        self.data_conf = data_conf
        self.d = self.data_conf["dimension"]
        self.num_classes = 100 
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
            print(len(self.Y))
            Y = self.Y[idcs] 
            return CustomTensorDataset(X=X,Y=Y,num_classes = self.num_classes, d=self.d,transform=self.transform)

    def build_dataset(self):
        data_dir  = self.data_conf['data_path']
        
        CIFAR100_TRAIN_MEAN = (0.5071, 0.4866, 0.4409)
        CIFAR100_TRAIN_STD = (0.2673, 0.2564, 0.2761)

        self.transform = transforms.Compose( [ transforms.RandomCrop(32, padding=4), 
                                                transforms.RandomHorizontalFlip(), 
                                                transforms.RandomRotation(15),
                                                transforms.ToTensor(),
                                                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)]
                                            )


        self.data = torchvision.datasets.CIFAR100(root=self.data_conf['data_path'], train=True,
                                        download=True, transform=self.transform)
        self.test_data  = torchvision.datasets.CIFAR100(root=self.data_conf['data_path'], train=False,
                                       download=True, transform=self.transform)
        self.X =  torch.tensor(self.data.data)
        
        #print(self.X[0])

        self.Y = torch.tensor(self.data.targets) 

        self.X_test =  torch.tensor(self.test_data.data)
        self.Y_test =  torch.tensor(self.test_data.targets)

        
    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return CustomTensorDataset(X=X_,Y=Y_, num_classes = self.num_classes,d=self.d,transform=None)
