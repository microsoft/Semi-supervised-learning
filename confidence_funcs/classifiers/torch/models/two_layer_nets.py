import torch 
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):

    def __init__(self,model_conf):
        super(TwoLayerNet, self).__init__()
        self.n_classes = model_conf["num_classes"]
        input_dim     = model_conf['input_dimension']
        hidden_dim    = model_conf['hidden_dim']
        activation    = model_conf['activation']
        
        self.layer1 =  nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, self.n_classes)
        if(activation=='relu'):            
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        self.classifier = nn.Sequential(self.layer1,self.activation,self.layer2)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x):
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        out = {}
        out['probs'] = probs 
        out['logits'] = logits
        out['abs_logits'] =  torch.abs(logits)
        return out
