import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GaussianClusters2D:
    def __init__(self,data_conf):
        self.data_conf = data_conf
    
    def build_dataset(self):
        np.random.seed(self.data_conf['seed'])    
        
        mu = self.data_conf['mu']
        sigma = self.data_conf['sigma']
        self.transform = None
        
        
        num_p = self.data_conf['n_samples_p']
        num_n = self.data_conf['n_samples_n']
        
        n = num_p + num_n
        X1 = np.random.multivariate_normal([mu,0],sigma*np.eye(2),num_p)
        X2 = np.random.multivariate_normal([-mu,0],sigma*np.eye(2),num_n)
        Y  = np.ones((n,1))
        Y[num_p:]=-1
        X  = np.vstack((X1,X2))
        X[:,0] = (X[:,0]-np.mean(X[:,0]))/np.std(X[:,0])
        X[:,1] = (X[:,1]-np.mean(X[:,1]))/np.std(X[:,1])
        
        Xy = np.hstack((X,Y))
        np.random.shuffle(Xy)  # does inplace shuffle
        Xy_train = Xy[:int(n*0.75)]
        
        self.X_train = Xy_train[:,[0,1]]
        self.Y_train = Xy_train[:,2]
        
        self.X_test = Xy[int(n*0.75):][:,[0,1]]
        self.Y_test = Xy[int(n*0.75):][:,2]
        
        self.train_instances = self.X_train
        self.train_labels = self.Y_train
        



    def getTrainLoader(self,batchSize=32):
        self.trainData = SubsetTensorDataset(torch.Tensor(self.train_instances),
                                   torch.Tensor(self.train_labels).long(),
                                   transform = None)
        return  DataLoader(self.trainData, batch_size=batchSize, shuffle=True, num_workers=1)
    
    def getTestLoader(self,batchSize=32):
                
        self.testData  = SubsetTensorDataset(torch.Tensor(X_test),
                                           torch.Tensor(Y_test).long(),
                                           transform=None)
        return  DataLoader(self.testData, batch_size=batchSize, shuffle=True, num_workers=1)
    
    def get_true_label(self,i):
        return self.train_labels[i]
    
    def get_traing_instance(self,i):
        return self.train_instances[i]
    
