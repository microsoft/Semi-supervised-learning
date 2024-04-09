import sys
sys.path.append('../')
sys.path.append('../../')

from .numpy_dataset import DatasetNumpy
import numpy as np
from ..dataset_utils import *

from src.utils.sampling_utils import *

class UniformUnitBallDataset(DatasetNumpy):
    def __init__(self,data_conf=None):
        self.data_conf = data_conf
        self.f = None 
     
    def build_dataset(self):
        self.transform = None

        self.num_classes = self.data_conf['num_classes']
        self.d = self.data_conf['dimension']
        n_train = self.data_conf['train_set_size']
        n_test = self.data_conf['test_set_size']
        
        X = np.array(random_ball(n_train+n_test,self.d))
        self.X = X[:n_train]
        

        if(self.data_conf['decision_boundary']=='quadratic'):
            
            A = np.array([[0.5,0],[0,0.5]])
            b = np.ones(2)
            b = b/np.linalg.norm(b)
            
            def f(x):
                if(np.linalg.norm(x)<=1 ):
                    return  np.dot(np.dot(x,A),x) + np.dot(x,b) - 0.3 
                else:
                    return None

            self.f = f

            z = np.array([np.dot(np.dot(X[i],A),X[i]) for i in range(len(X))])  + np.dot(X,b) - 0.3
            Y = np.array([1 if z[i]>0 else 0 for i in range(len(z)) ]) 
            

        else:
            
            w = np.ones(self.d)
            w = w/np.linalg.norm(w)

            Y = np.array([1 if np.dot(w,x)>= 0 else 0 for x in X])

            def f(x):
                if(np.linalg.norm(x)<=1 ):
                    return np.dot(w,x)
                else:
                    return None
                  
            self.f = f 
        
        self.Y = Y[:n_train]

        self.test_set = DatasetNumpy(X[n_train:],Y[n_train:],self.num_classes,f)
        
    def get_test_datasets(self):
        return self.test_set
            
        
        

    

