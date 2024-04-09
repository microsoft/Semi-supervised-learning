from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

from datasets.dataset_utils import *
from datasets.numpy_dataset import DatasetNumpy

from torch.utils.data import  Dataset
import numpy as np
from PIL import Image

class MNISTData_sklearn(DatasetNumpy):
    def __init__(self,data_conf):
        self.data_conf = data_conf
        self.f = None

    def build_dataset(self):
        data_dir  = self.data_conf['data_path']
        self.data = MNIST(data_dir, download=True,
                                transform=None)
        
        self.testData  = MNIST(data_dir, train=False, download=True,
                               transform=None)
        
        self.X =  self.data.data.numpy()
        self.Y =  self.data.targets.numpy()

        self.X_test =  self.testData.data.numpy()
        self.Y_test =  self.testData.targets.numpy()
        
        if "flatten" in self.data_conf.keys() and self.data_conf['flatten'] == True: # flat the image from 28x28 to 784
            self.X =  self.X.reshape(self.X.shape[0],-1)
            self.X_test =  self.X_test.reshape(self.X_test.shape[0],-1)

    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return DatasetNumpy(X=X_,Y=Y_,f=self.f)