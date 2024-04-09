import sys
from .numpy_dataset import DatasetNumpy
import numpy as np
from datasets.dataset_utils import *

from utils.sampling_utils import *

class XORBallsDataset(DatasetNumpy):
    def __init__(self,data_conf=None):
        self.data_conf = data_conf
        self.num_classes = data_conf['num_classes']
        self.f = None 
     
    def build_dataset(self):
        conf = self.data_conf 
        self.transform = None

        self.d = self.data_conf['dimension']
    
        #w = np.ones(self.d)
        #w = w/np.linalg.norm(w)
        w = np.array([ 0.70814278, -0.70606926])
        b =  1.44712748


        s = 2
        mu1 = np.array([1,1])*s 
        mu2 = np.array([-1,1])*s 
        mu3 = np.array([-1,-1])*s 
        mu4 = np.array([1,-1])*s  
        n_train = self.data_conf['train_set_size']
        n_test = self.data_conf['test_set_size']
        n = n_train + n_test 
        
        n1 = n//4
        X1 = random_ball(n1,2) + mu1 
        X2 = random_ball(n1,2) + mu2 
        X3 = random_ball(n1,2) + mu3
        X4 = random_ball(n1,2) + mu4 

        X= np.vstack((X1,X3,X2,X4)) 
        num_p = len(X1)+len(X3) 
        num_n = len(X2) + len(X4)
        n = num_p + num_n
        Y = np.zeros((n,1))
        Y[:n,0] = np.hstack( ( np.array([1]*num_p), np.array([0]*num_n)))
        Y = Y.astype(int)
        Xy = np.hstack((X,Y)) 
        
        np.random.shuffle(Xy)

        self.X = Xy[:n_train,[0,1]]
        self.Y = Xy[:n_train,2].astype(int)

        self.test_set = DatasetNumpy(Xy[n_train:,[0,1]], Xy[n_train:,2].astype(int),num_classes=self.num_classes)

        def f(x):
            return np.dot(w,x) + b
                
        self.f = f 

    def get_test_datasets(self):
        return self.test_set
        
        