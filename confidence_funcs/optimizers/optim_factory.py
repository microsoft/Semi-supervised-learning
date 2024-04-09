
import torch
from torch import nn 
from torch import optim

from .sam import SAM

def get_optimizer(model,train_params):
    opt_name = train_params['optimizer']
    
    optimizer = None 
    lr_scheduler = None 
    params = None 
    if( isinstance(model,torch.nn.parameter.Parameter)):
        params = [model]
    else:
        params = model.parameters()

    if(opt_name=='adam'):      
        optimizer = optim.Adam(params, 
                                lr            =  train_params['learning_rate'],
                                weight_decay  =  train_params['weight_decay']
                                )
    elif(opt_name == 'lbfgs'):
        optimizer = torch.optim.LBFGS(params, 
                                        lr = train_params['learning_rate'])
    elif(opt_name == 'sam'):
        base_optimizer = torch.optim.SGD
        optimizer =  SAM(params, base_optimizer, lr=train_params['learning_rate'], momentum=train_params['momentum'], weight_decay=train_params['weight_decay'])
    else:
        optimizer = optim.SGD(params, 
                                lr            = train_params['learning_rate'], 
                                momentum      = train_params['momentum'], 
                                weight_decay  = train_params['weight_decay'],
                                nesterov      = train_params['nesterov'])

    if(train_params['use_lr_schedule']):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    return optimizer, lr_scheduler
    
