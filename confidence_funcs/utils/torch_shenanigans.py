
import torch

def safe_squeeze(x):
    s = x.shape 
    m = max(s)
    if(len(x)>1):
        return x.squeeze() 
    
    elif(len(s)>1 and m==1):
        return x[0]
    else:
        return x 
    
def model_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm

    