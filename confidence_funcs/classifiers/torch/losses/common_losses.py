from  .abstract_loss import * 
import torch 
from torch import nn 
import torch.nn.functional as F  

class StdLoss(AbstractLoss):
    def __init__(self):
        super(StdLoss, self).__init__()
        

    def batch_closure_callback(self, batch_state=None):
        pass
    
    def epoch_closure_callback(self, epoch_state=None ):
        pass


class StdCrossEntropyLoss(StdLoss):

    def __init__(self,loss_conf={'label_smoothing':0.0}):
        super(StdCrossEntropyLoss, self).__init__()
        if('label_smoothing' in loss_conf):
            self.xent = nn.CrossEntropyLoss(label_smoothing=loss_conf['label_smoothing'])
        else:
            self.xent = nn.CrossEntropyLoss()
        
        self.result = {}

    # input is assumed to be logits and output is labels.
    def forward(self,input,target,idx=None):
        loss  = self.xent(input, target) 
        self.result['loss'] = loss.item()
        return loss 
    

        
class StdSquentropyLoss(StdLoss):
    def __init__(self, loss_conf={}):
        super(StdSquentropyLoss, self).__init__()
        self.xent = nn.CrossEntropyLoss()
        self.result = {}
    
    def forward(self,input, target, idx=None ):
        # input is n_b x k : where n_b is batch size and k is num of classes
        k = input.shape[1] 
        xent  = self.xent(input, target)  
        reg   = torch.mean((torch.mul(input, 1-F.one_hot(target,num_classes=k)))**2)
        loss =  xent + reg 

        self.result['xent']  = xent.item()
        self.result['reg']   = reg.item()
        self.result['loss']  = loss.item()
        
        return loss 


# implement more standard losses as needed...

