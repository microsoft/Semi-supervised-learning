import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  

class PyTorchLogisticRegression(nn.Module):

    def __init__(self,model_conf,logger=None):
        
        super().__init__()
        
        self.linear_layer = nn.Linear(model_conf['input_dimension'],1, bias=model_conf['fit_intercept']) 

        if(logger is not None):
            self.logger = logger 
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x):

        logit0 = -self.linear_layer(x)
        logit1 = -logit0 
        logits = torch.column_stack([logit0,logit1])
        probs = F.softmax(logits, dim=1)
        out = {} 
        
        out['probs'] = probs 
        out['abs_logits'] =  torch.abs(logits)
        out['logits'] = logits 

        return out 

    def normalize_weights(self):
        w = self.linear_layer.weight
        w = w/torch.norm(w)
        self.linear_layer.weight = torch.nn.Parameter(w)
        #w = self.linear_layer.weight
        #print(torch.norm(w))

    #def scale_weights(self,c):
        #self.linear_layer.weight = self.linear_layer.weight * c 


    
    