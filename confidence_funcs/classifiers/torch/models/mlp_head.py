import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  






class MLPHeadOld(nn.Module):
    def __init__(self, model_conf, bb_model=None):
        super(MLPHeadOld, self).__init__()

        in_dim = model_conf['input_dimension']
        out_dim = model_conf['num_classes']

        #self.bb_model = bb_model  #backbone model for features
        
        #self.bb_model.disable_grads()
        
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim*20),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Linear(1000, 1000),
            #nn.Tanh(),
            #nn.Linear(1000, 1000),
            #nn.Tanh(),
            nn.Linear(in_dim*20, in_dim*20),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Linear(1500, 1000),
            #nn.Tanh(),
            nn.Linear(in_dim*20, in_dim*20),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(in_dim*20, in_dim*10),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(in_dim*10, in_dim),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            #nn.ReLu()
            nn.Linear(out_dim, 1)

            #nn.Tanh(),
            #nn.Linear(1000, 500),
            #nn.Tanh(),
            #nn.Linear(500, 100),
            #nn.Tanh(),
            
        )
        
        '''
        self.head = nn.Linear(in_dim,1)
        '''

        '''
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            #nn.Tanh(),
            nn.Tanh(),
            #nn.Linear(1000, 1000),
            #nn.Tanh(),
            #nn.Linear(1000, 1000),
            #nn.Tanh(),
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),

            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            #nn.ReLu()
            nn.Linear(out_dim, 1)

            #nn.Tanh(),
            #nn.Linear(1000, 500),
            #nn.Tanh(),
            #nn.Linear(500, 100),
            #nn.Tanh(),
            
        )
        
        '''
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        
        #bb_out = self.bb_model(x)

        #z = bb_out['logits']
        
        out = self.head(x)
        
        probs = F.softmax(out)
        output = {}
        output['probs'] = probs 
        output['abs_logits'] =  torch.abs(out)
        output['logits'] = out 

        return output