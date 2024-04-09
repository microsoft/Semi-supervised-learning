from abc import ABC, abstractmethod
import torch.nn as nn 

class AbstractLoss(ABC, nn.Module ):
    def __init__(self):
        super(AbstractLoss, self).__init__()
    
    @abstractmethod
    def forward(self,input,target,idx):
        pass
    
    @abstractmethod
    def batch_closure_callback(self, batch_state):
        pass
    
    @abstractmethod
    def epoch_closure_callback(self, epoch_state):
        pass

