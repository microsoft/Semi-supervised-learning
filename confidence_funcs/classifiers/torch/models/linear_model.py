import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class LinearModel(nn.Module):

    def __init__(self,model_conf):
        super(LinearModel, self).__init__()

        input_dim     = model_conf['input_dimension']
        num_classes   = model_conf['num_classes']
        fit_intercept = model_conf['fit_intercept']

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=input_dim, out_features=num_classes, bias=fit_intercept) 
        #by default bias is True
        #self.act1 = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits  = self.fc1(x)
        #logits  = self.act1(logits)
        probs = F.softmax(logits,dim=1)
        #print(logits.shape,probs.shape)
        out = {}
        out['probs'] = probs 
        out['logits'] = logits
        
        out['abs_logits'] =  torch.abs(logits)

        return out



    def embedding(self,x):
        logits  = self.fc1(x)
        probs = F.softmax(logits)
        out = {}
        out['embedding'] = logits.detach().numpy()
        probs.mean().backward()
        out['grad_embedding'] = probs
        return out

    def criterion(self,input,targets):
        loss = nn.CrossEntropyLoss()
        return loss(input,targets)