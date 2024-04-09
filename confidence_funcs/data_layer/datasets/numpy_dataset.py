
import numpy as np 

class DatasetNumpy:

    def __init__(self,X=None,Y=None,num_classes=2,f=None):
        self.X = X 
        self.Y = Y
        self.d = X.shape[1] 
        self.f = f
        self.num_classes = num_classes
    
    def build_dataset(self):
        pass 
    
    def len(self):
        return len(self.X)
        
    def __len__(self):
        return len(self.X) 
    
    def get_subset(self,idcs,Y_=None):
        '''
           Y_ must have same size as idcs and Y_[i] = label of idcs[i].
           If this Y_ is given that means the labels of the subset are to 
           be populated from it, else populate the labels from original labels 
           if available.
        '''
        idcs = np.array(idcs) 
        X_ = None 
        if(self.X is not None):
            X_ = self.X[idcs]

        if(Y_ is None and self.Y is not None):
            Y_ = self.Y[idcs]
        
        return DatasetNumpy(X=X_,Y=Y_,f=self.f)

    def get_decision_boundary(self,th=1e-3,t=1000,x1_lim=[-1,1],x2_lim=[-1,1]):
        
        x = np.linspace(x1_lim[0],x1_lim[1],t)
        y = np.linspace(x2_lim[0],x2_lim[1],t)
        z = np.meshgrid(x,y)
        X_ = np.array(list(zip(*(c .flat for c in z))))
        F = [self.f(X_[i]) for i in range(len(X_))]

        z = np.array([X_[i] for i in range(len(X_)) if  F[i] is not None and  abs(F[i]) <= th ])
        
        return z

    def get_random_fraction(self,frac=1.0):
        n = len(self.X)
        idcs = np.random.choice(n,size = int(frac*n),replace=False) 
        return self.get_subset(idcs)



        
    
    
 