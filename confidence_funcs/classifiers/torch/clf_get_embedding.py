from asyncio.log import logger
import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader
import numpy as np

class ClassifierEmbedding:
    def __init__(self,logger):
        self.logger = logger
    
    def get_embedding(self,model,dataset,inference_conf={}):
        
        if(inference_conf is None):
            inference_conf = {}
        inference_conf.setdefault('batch_size',64)
        inference_conf.setdefault('shuffle',False)
        inference_conf.setdefault('device','cpu')
        
        device = inference_conf['device']
        data_loader = DataLoader(dataset=dataset,batch_size= inference_conf['batch_size'], shuffle=inference_conf['shuffle'])

        model = model.to(device)

        self.logger.info("Running get embedding on {}, batch size {}".format(device,inference_conf['batch_size']))
        with torch.no_grad():
            model.eval() 
            lst_embedding = []
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                embedding  = model.get_embedding(data)['embedding']
                lst_embedding.extend(embedding.cpu().numpy())
            out = {} 
            out['embedding'] = torch.Tensor(np.array(lst_embedding))
            return out 

    def get_gard_embedding(self,model,dataset,inference_conf={}):
        
        if(inference_conf is None):
            inference_conf = {}
        inference_conf.setdefault('batch_size',64)
        inference_conf.setdefault('shuffle',False)
        inference_conf.setdefault('device','cpu')
        
        device = inference_conf['device']
        data_loader = DataLoader(dataset=dataset,batch_size= inference_conf['batch_size'], shuffle=inference_conf['shuffle'])

        model = model.to(device)
        
        with torch.no_grad():
            model.eval() 
            lst_embedding = []
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                embedding  = model.get_grad_embedding(data)
                lst_embedding.extend(embedding.cpu().numpy())
            out = {} 
            out['grad_embedding'] = torch.Tensor(np.array(lst_embedding)) 
            return out 

            
