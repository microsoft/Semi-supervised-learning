from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from datasets.dataset_utils import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


class GeneralSynthetic:
    def __init__(self,conf):
        self.conf = conf
    
    def build_dataset(self):
        
        data_conf = self.conf['data_conf']

        self.transform = None
        
        
        X,Y = make_classification(n_samples    = data_conf['n_samples'], 
                                  n_features   = data_conf['n_features'],
                                  n_informative= data_conf['n_informative'], 
                                  n_redundant  = data_conf['n_redundant'],
                                  n_repeated   = data_conf['n_repeated'], 
                                  n_classes    = data_conf['n_classes'], 
                                  n_clusters_per_class=data_conf['n_clusters_per_class'],
                                  class_sep    = data_conf['class_sep'],
                                  flip_y       = data_conf['flip_y'],
                                  weights      = data_conf['weights'], 
                                  random_state = data_conf['random_state'])
        
        
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=data_conf['test_size']
                                                            ,random_state=data_conf['random_state'] 
                                                            )
        self.X_train = X_train 
        self.Y_train = Y_train 
        self.X_test = X_test 
        self.Y_test = Y_test 

        if(data_conf['tensorize']):
            self.train_dataset = CustomTensorDataset(torch.Tensor(self.X_train),
                                           torch.Tensor(self.Y_train).long(),
                                           transform = None)
        
            self.test_dataset  = CustomTensorDataset(torch.Tensor(X_test),
                                           torch.Tensor(Y_test).long(),
                                           transform=None)
    
    

        
   
