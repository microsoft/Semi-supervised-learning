import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  

class MLP(nn.Module):
    def __init__(self,model_conf):
        super(MLP, self).__init__()
        in_dim = model_conf['input_dimension']
        out_dim = model_conf['num_classes']

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 1000),
            nn.Tanh(),
            nn.Linear(1000, 500),
            nn.Tanh()
            #nn.Linear(1000, 1000),
            #nn.Tanh(),
            #nn.Linear(2000, 1500),
            #nn.Tanh(),
            #nn.Linear(1500, 1000),
            #nn.Tanh(),
            #nn.Linear(1500, out_dim)
            #nn.Tanh(),
            #nn.Linear(1000, 500),
            #nn.Tanh(),
            #nn.Linear(500, 100),
            #nn.Tanh(),
        )

        #self.pre_last_layer = nn.Linear(2000, 1500)
        u = int(1.5*out_dim)
        self.pre_last_layer = nn.Linear(500, u )
            
        self.pre_last_act   =    nn.Tanh()
            
        self.last_layer = nn.Linear(u, out_dim) 

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output = {}
        x = self.layers(x)
        x = self.pre_last_act(self.pre_last_layer(x))

        output['pre_logits']  = x 

        x = self.last_layer(x)

        probs = F.softmax(x,dim=1)
        
        
        output['probs'] = probs 
        output['abs_logits'] =  torch.abs(x)
        output['logits'] = x 

        return output