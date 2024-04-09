import torch
from torch.utils.data import DataLoader
import numpy as np

class ClassfierInference:
    def __init__(self,logger):
        self.logger = logger
    
    def predict(self,model,dataset,inference_conf={}): 
        
        if(inference_conf is None):
            inference_conf = {}
            inference_conf['device'] = str(next(model.parameters()).device)  
            
        inference_conf.setdefault('batch_size',256)
        inference_conf.setdefault('shuffle',False)
        
        #inference_conf.setdefault('device','cpu')
        #self.logger.info("Running inference on {}".format(inference_conf['device']))

        device = inference_conf['device']
        data_loader = DataLoader(dataset=dataset,
                                  batch_size= inference_conf['batch_size'], 
                                  shuffle=inference_conf['shuffle'],
                                  pin_memory=True, 
                                  num_workers=2)
        model = model.to(device)
        with torch.no_grad():
            model.eval() 
            lst_confs = []
            lst_probs = []
            lst_preds = []
            lst_energy_scores = []
            lst_abs_logits = []
            lst_all_logits = []
            lst_all_pre_logits = []
            T = 1.0
            for batch_idx, (data, target, idx) in enumerate(data_loader):
                if isinstance(data,torch.Tensor):
                    data = data.to(device)
                if isinstance(target,torch.Tensor):
                    target = target.to(device)
                out  = model.forward(data)
                probs = out['probs']
                confidence, y_hat = torch.max(probs, 1)
                lst_confs.extend(confidence.cpu().numpy())
                lst_preds.extend(y_hat.cpu().numpy())
                lst_probs.extend(probs.cpu().numpy())
                lst_all_logits.append(out['logits'].cpu().numpy())
                if('pre_logits' in out):
                    lst_all_pre_logits.append(out['pre_logits'].cpu().numpy())

                lst_energy_scores.extend(-T*torch.logsumexp(out['logits'] / T, dim=1).cpu().numpy())

                if('abs_logits' in out):
                    abs_logits, idcs = torch.max(out['abs_logits'],1)
                    lst_abs_logits.extend(abs_logits.cpu().numpy()) 
            
            out = {} 
            out['labels'] = torch.Tensor(np.array(lst_preds) ).long()
            out['confidence'] = torch.Tensor(np.array(lst_confs))#confidence  # 1d array
            out['probs'] = torch.Tensor(np.array(lst_probs) ) # n\times k ( k classes)
            out['logits'] = torch.Tensor(np.vstack(lst_all_logits))
            if(len(lst_all_pre_logits)>0):
                out['pre_logits'] = torch.Tensor(np.vstack(lst_all_pre_logits))

            # make it positive to be consistent with other scores.
            out['energy_score'] = -torch.Tensor(np.array(lst_energy_scores)) 
            
            if(len(lst_abs_logits)>0):
                out['abs_logit'] = np.array(lst_abs_logits ) # 1d array
            return out 

            
