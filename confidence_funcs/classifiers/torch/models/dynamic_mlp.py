
import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  


'''
Example config 
model_conf = {}
model_conf['input_dimension'] = 2
model_conf['output_dimension'] = 2 

model_conf['layers'] = [
    { "type":"linear", "dim_factor":2},
    { "type": "activation", "act_fun": "relu"},
    {  "type":"linear", "dim_factor":2},
    { "type": "activation", "act_fun": "relu"},
]
'''
class DynamicMLP(nn.Module):
    def __init__(self, model_conf):
        super(DynamicMLP, self).__init__()

        self.in_dim  = model_conf['input_dimension']
        self.out_dim = model_conf['output_dimension']
        #k       = model_conf['num_classes']
        layers = nn.ModuleList()

        if('layers' not in model_conf or model_conf['layers'] is None or len(model_conf['layers'])==0):
            layers.add_module('layer_0', nn.Linear(self.in_dim,self.out_dim))

        elif( 'layers' in model_conf and  len(model_conf['layers']) > 0):
            p_dim = [self.in_dim]

            for i in range(len(model_conf['layers'])):
                l_conf = model_conf['layers'][i]
                layer = None                 
                if(l_conf['type']=='linear'):
                    d2 = int(self.in_dim * l_conf['dim_factor'])
                    layer =  nn.Linear(  p_dim[i], d2)
                    p_dim.append(d2)

                elif(l_conf['type'] == 'activation'):
                    if(l_conf['act_fun']=='relu'):
                        layer =  nn.ReLU() 
                    elif(l_conf['act_fun']=='tanh'):
                        layer = nn.Tanh()
                    p_dim.append(p_dim[i])

                layers.add_module( f'layer_{i}' , layer)
            
            layers.add_module('last_layer', nn.Linear( p_dim[-1], self.out_dim) )


        self.net = nn.Sequential(*layers)

    def forward(self,x ):
        
        out = self.net(x)
        
        probs = F.softmax(out, dim=1)
        output = {}
        output['probs'] = probs 
        output['abs_logits'] =  torch.abs(out)
        output['logits'] = out 
        return output 