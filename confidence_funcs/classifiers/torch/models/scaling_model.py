import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TemperatureScalingModel(nn.Module):

    def __init__(self,model_conf):
        super(TemperatureScalingModel, self).__init__()
        
        # an affine operation: y = x/t
        self.temperature = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):

        logits  = x/self.temperature + self.bias

        #logits  = self.act1(logits)
        probs = F.softmax(logits,dim=1)
        
        out = {}
        out['probs'] = probs 
        out['logits'] = logits
        
        out['abs_logits'] =  torch.abs(logits)

        return out
    