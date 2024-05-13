import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

class ClassfierInference:
    def __init__(self,logger):
        self.logger = logger
    
    def predict(self,model,dataset,inference_conf={}): 
        # pass these two in inference_conf
        # feature_key "x_lb"  "x_ulb", "x_ulb_w" etc.
        # idx_key : "idx_lb" , "idx_ulb"

        feature_key = inference_conf['feature_key']
        idx_key     = inference_conf['idx_key']
                
        if('device' not in inference_conf):
            inference_conf['device'] =  str(next(model.parameters()).device)  

        inference_conf.setdefault('batch_size',1024)
        inference_conf.setdefault('shuffle',False)
        
        #inference_conf.setdefault('device','cpu')
        #self.logger.info("Running inference on {}".format(inference_conf['device']))
        
        device = inference_conf['device']
        data_loader = DataLoader(dataset=dataset,
                                  batch_size= inference_conf['batch_size'], 
                                  shuffle=inference_conf['shuffle'],
                                  pin_memory=True, 
                                  num_workers=4)
        
        print(inference_conf['device'],inference_conf['batch_size'])

        model = model.to(device)
        n = len(dataset)
        
        k = dataset.num_classes

        with torch.no_grad():
            model.eval() 

            out = {} 
            out['labels'] = torch.zeros(n).long().to(device)
            out['confidence'] = torch.zeros(n).to(device) #confidence  # 1d array
            out['probs'] = torch.zeros(n,k).to(device) # n\times k ( k classes)
            out['logits'] = torch.zeros(n,k).to(device)
            #print(out['probs'].shape)
            lst_all_pre_logits = []
            j=0
            for i, data_dict in enumerate(data_loader):
                #data   = data_dict['x_lb'] if 'x_lb' in data_dict else data_dict['x_ulb_w']
                data = data_dict[feature_key]
                idx  = data_dict[idx_key]
                #target = data_dict['y_lb'] 

                if isinstance(data,torch.Tensor):
                    data = data.to(device)
                    idx  = idx.to(device)
                
                out_batch  = model.forward(data)
                probs = F.softmax(out_batch['logits'], dim=1)
                
                bs = len(idx)
                
                
                confidence, y_hat = torch.max(probs, 1)
                
                out['probs'][j:j+bs][:] = probs 
                out['confidence'][j:j+bs] = confidence 
                out['labels'][j:j+bs] = y_hat 
                out['logits'][j:j+bs][:] = out_batch['logits']

                
                j+=bs

                if('pre_logits' in out_batch):
                    lst_all_pre_logits.append(out_batch['pre_logits'].cpu().numpy()) 

            if(len(lst_all_pre_logits)>0):
                out['pre_logits'] = torch.Tensor(np.vstack(lst_all_pre_logits)).to(device)

            return out 

            
