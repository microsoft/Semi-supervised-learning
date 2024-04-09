from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

from ..dataset_utils import *


from torch.utils.data import  Dataset
import numpy as np
from PIL import Image

class MNISTData(Dataset):
    def __init__(self,data_conf):
        self.data_conf = data_conf
        self.d = self.data_conf["dimension"]
        self.num_classes = 10 
        self.f = None
        
    def __getitem__(self, index):
        
        x = self.X[index] #.float()
        
        #if self.transform:
            #x = Image.fromarray(x.numpy().astype(np.float))
            #x = Image.fromarray(x.numpy())
        #    x = self.transform(x)
        
        y = self.Y[index]
        
        return x, y
    
    def __len__(self):

        return len(self.X)
    
    def len(self):
        return len(self.X)
    '''
    def get_subset(self,idcs):
        idcs = np.array(idcs) 
        X = None
        Y = None 
        if(self.X is not None):
            X = self.X[idcs] 
        if(self.Y is not None):
            Y = self.Y[idcs]
            return CustomTensorDataset(X=X,Y=Y,num_classes = self.num_classes, d=self.d,transform=self.transform)
    '''

    def get_subset(self,idcs,Y_=None):
        '''
           Y must have same size as idcs and Y[i] = label of idcs[i].
           If this Y is given that means the labels of the subset are to 
           be populated from it, else populate the labels from original labels 
           if available.
        '''
        idcs = np.array(idcs) 
        X_ = None
        Y_ = None

        # check if the type of self.x is a numpy array
        if(type(self.X) == type(np.array([]))): 
            if(self.X is not None):
                X_ = self.X[idcs] 
            if(Y_ is None and self.Y is not None):
                Y_ = self.Y[idcs]
        else:
            if(self.X is not None):
                X_ = [self.X[i] for i in idcs] 
            if(self.Y is not None):
                Y_ = tuple([self.Y[i] for i in idcs] )
            
            Y_ = torch.Tensor(Y_).long()
            
            if(len(X_)>0):
                X_ = torch.stack(X_)

        return CustomTensorDataset(X=X_,Y=Y_,transform=None)

    def build_dataset(self):
        data_dir  = self.data_conf['data_path']
        
        #self.transform = transforms.Compose([
        #                        transforms.Resize((32, 32)),
        #                        transforms.ToTensor(),
        #                        transforms.Normalize((0.1307,), (0.3081,))
        #                    ])
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor()])

        self.data = MNIST(data_dir, download=True,
                                transform=self.transform)
        
        #self.data.data = self.data.data.float()
        
        #print(self.data.data.shape)
        #print(type(self.data.data))

        self.test_data  = MNIST(data_dir, train=False, download=True,
                               transform=self.transform)
        
        self.X =  self.data.data
        self.Y =  self.data.targets

        #self.test_data.data = self.test_data.data.float()
        self.X_test =  self.test_data.data
        
        self.Y_test =  self.test_data.targets
        
        if "flatten" in self.data_conf.keys() and self.data_conf['flatten'] == True: # flat the image from 28x28 to 784
            
            self.X =  self.X.reshape(self.X.shape[0],-1).float()
            self.X_test =  self.test_data.data.reshape(self.test_data.data.shape[0],-1).float()

            self.transform = None
        
    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return CustomTensorDataset(X=X_,Y=Y_, num_classes = self.num_classes,d=self.d,transform=self.transform)
