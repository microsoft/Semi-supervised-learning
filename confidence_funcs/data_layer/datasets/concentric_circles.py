import sys
from .numpy_dataset import DatasetNumpy
import numpy as np
from ..dataset_utils import *
from src.utils.sampling_utils import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

class ConcentricCircles(DatasetNumpy):
    def __init__(self,data_conf):
        self.data_conf = data_conf
        data_conf.setdefault('noise',0.05)
        self.f = None 
        
        
    def build_dataset(self):
        
        data_conf = self.data_conf
        #dataDir  = data_conf['data_path']
        
        self.transform = None

        self.num_classes = self.data_conf['num_classes']
        self.d = self.data_conf['dimension']
        n_train = self.data_conf['train_set_size']
        n_test = self.data_conf['test_set_size']

        n_samples = n_train + n_test 
        noise = data_conf['noise']

        X,Y = make_circles(n_samples=n_samples, factor=0.5, noise=noise)        

        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=data_conf['test_set_size']
                                                            ,random_state=data_conf['random_state'] )

        
        self.X = X_train 
        self.Y = Y_train 

        self.X_test = X_test 
        self.Y_test = Y_test 
        self.test_set = DatasetNumpy(X_test,Y_test,self.num_classes,None)

    def get_test_datasets(self):
        return self.test_set 
