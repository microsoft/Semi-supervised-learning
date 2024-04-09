'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  .abstract_loss import * 

class FocalLoss(AbstractLoss):
    def __init__(self, loss_conf=None):

        super(FocalLoss, self).__init__()

        self.gamma = loss_conf['gamma']
        self.size_average = loss_conf['size_average']

        self.result = {}

    def forward(self, input, target ,idx=None ):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        x = input.dim()
        input_dimension = 0 if x == 0 or x == 1 or x == 3 else 1
        logpt = F.log_softmax(input, dim = input_dimension)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt 
        if self.size_average: 
            loss =  loss.mean()
        else: 
            loss =  loss.sum()

        self.result["loss"] = loss.item() 

        return loss 
    
    def batch_closure_callback(self, batch_state):
        pass 
    
    def epoch_closure_callback(self, epoch_state):
        pass 

'''
Implementation of Focal Loss with adaptive gamma.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.special import lambertw
import numpy as np


def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma


class FocalLossAdaptive(AbstractLoss):
    def __init__(self, loss_conf= {"gamma":0,"size_average":False, "device":"cuda:0"}):
        super(FocalLossAdaptive, self).__init__()

        self.size_average = loss_conf["size_average"]
        self.gamma =  loss_conf["gamma"]
        self.device =  loss_conf["device"]

        ps = [0.2, 0.5]
        gammas = [5.0, 3.0]
        i = 0
        self.gamma_dic = {}
        for p in ps:
            self.gamma_dic[p] = gammas[i]
            i += 1
    
    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(self.gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(self.gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target, idx=None ):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        
        gamma = self.get_gamma_list(pt)
        
        loss = -1 * (1-pt)**gamma * logpt

        if self.size_average: 
            loss =  loss.mean()
        else: 
            loss =  loss.sum()
            
        self.result["loss"] = loss.item() 

        return loss 
    
    
    def batch_closure_callback(self, batch_state):
        pass
    
    
    def epoch_closure_callback(self, epoch_state):
        pass

        