import sys 
sys.path.append('../')
sys.path.append('../../')

import torch
from .abstract_calibrator import AbstractCalibrator

from src.models.torch.optimizers import get_optimizer 
from src.models.torch.mlp_head import * 
from src.datasets.dataset_utils import * 
from src.utils.torch_shenanigans import * 


class AutoLabelingModel(nn.Module):
    def __init__(self, model_conf):
        super(AutoLabelingModel,self).__init__()

        self.g_model =  MLPHead(model_conf)
        self.t       =  torch.nn.Parameter(torch.tensor([0.95]))
    

class AutoLabelingOptimization_V1(AbstractCalibrator):

    def __init__(self,clf, calib_conf,logger):

        self.calib_conf = calib_conf
        # the classifier model for which we are learning g and t
        self.cur_clf = clf  
        self.logger = logger 

    def init(self,ds_calib, clf_inf_out=None):

        calib_conf    = self.calib_conf 
        cur_clf       = self.cur_clf

        #train_conf    = self.calib_conf.training_conf
        train_conf_g    = self.calib_conf.training_conf_g
        train_conf_t    = self.calib_conf.training_conf_t
        
        auto_lbl_conf = calib_conf.auto_lbl_conf

        if(clf_inf_out is None):
            clf_inf_out = cur_clf.predict(ds_calib)

        features    = clf_inf_out['logits']

        self.eps    = auto_lbl_conf.auto_label_err_threshold

        self.C_1    = auto_lbl_conf.C_1


        self.Y_correct = (clf_inf_out['labels'] != ds_calib.Y).long()
        # 1 ==> incorrect prediction (mistake), 0 ==> correct prediction (no mistake)

        
        # create dataset with featuers and Y_correct labels 
        self.ds_calib_train = CustomTensorDataset(X= features,Y=self.Y_correct)

        if("model_conf" not in calib_conf):
            g_model_conf = {} 
        else:
            g_model_conf = calib_conf.model_conf
        
        if("input_dimension" not in g_model_conf):
            g_model_conf["input_dimension"] = features.shape[1]

        if("num_classes" not in g_model_conf):
            g_model_conf["num_classes"] = features.shape[1]
        

        self.auto_model     = AutoLabelingModel(g_model_conf)

        
        self.optimizer_g, self.lr_scheduler_g = get_optimizer(self.auto_model,train_conf_g)
        self.optimizer_t, self.lr_scheduler_t = get_optimizer(self.auto_model,train_conf_t)


    def fit(self,calib_ds,calib_ds_inf_out=None):
        
        self.init(calib_ds,calib_ds_inf_out)
        logger = self.logger 

        calib_conf    = self.calib_conf
        device        = calib_conf['device']
        #train_conf    = self.calib_conf.training_conf

        epochs = self.calib_conf.training_conf_g['max_epochs']

        logger.debug(self.calib_conf.training_conf_g)
        logger.debug(self.calib_conf.training_conf_t)

        self.auto_model.train()
        self.auto_model = self.auto_model.to(device)

        g_model = self.auto_model.g_model
        t       = self.auto_model.t 

        D = {"err_":[],"cov_":[], "err":[], "cov":[], "loss":[], "S":[],'A':[],'B':[]}
        
        logger.debug('Using optimizer for g: {}'.format(type(self.optimizer_g)))
        logger.debug('Using optimizer for t: {}'.format(type(self.optimizer_t)))


        for epoch in range(epochs):
            
            self.optimizer_g.zero_grad()
            self.optimizer_t.zero_grad()
            
            lst_cov_ = []
            lst_err_ = []
            lst_S = []
            lst_targets = []
            lst_g = []

            for inputs, targets,idx in DataLoader( self.ds_calib_train, batch_size=64):
                
                #inputs.requires_grad_(True)
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(targets,torch.Tensor):
                    targets = targets.to(device)

                out        = g_model.forward(inputs)
                logits     = safe_squeeze(out['logits'])


                #g          = torch.abs(logits) 
                g = logits

                S          = torch.clamp(torch.sigmoid(g-t),0,1)

                lst_S.append(S)
                lst_g.append(g)

                lst_targets.append(targets)

            
            S       = torch.cat(lst_S,dim=0)
            targets = torch.cat(lst_targets,dim=0)
            n       = len(S)

            cov_surrogate    = torch.sum(S)/n
            
            err_surrogate    = torch.dot(S, targets.float())/torch.sum(S)
            l1 = self.calib_conf['l1']
            l2 = self.calib_conf['l2']

            
            
            #loss = -cov_surrogate*l1 + l2*(err_surrogate- self.eps)*(err_surrogate- self.eps)
            u = self.C_1 * torch.sqrt(err_surrogate * (1-err_surrogate))

            loss = -cov_surrogate*l1 + l2*torch.abs((err_surrogate + u -  self.eps ))
            #loss = -cov_surrogate*0.01 + err_surrogate #torch.abs((err_surrogate- self.eps))
            #loss = -cov_surrogate*l1 + err_surrogate*l2 #torch.abs((err_surrogate- self.eps))
            G = torch.cat(lst_g,dim=0)

            loss.backward() 

            logger.debug(f'Loss: {loss.item()}')


            self.optimizer_g.step()

            self.optimizer_t.step()

            self.lr_scheduler_g.step()
            self.lr_scheduler_t.step()


            
            logger.debug(f'Epoch: {epoch} Loss :{loss.item()}')

            # for logging and checking progress
            U       =  S>=0.5
            A       =  targets==0   # where no error is made 
            B       =  targets==1   # where error is made
            
            auto_err = torch.sum(torch.logical_and(U,B))/(torch.sum(U)+1)
            cov      = (torch.sum(U))/n 
            
            logger.debug(f"Epoch: {epoch} Surrogate Coverage : {cov_surrogate.item()} Surrogate Error: {err_surrogate.item()} ")
            
            logger.debug(f'Epoch: {epoch} At t = 0.5, {epoch} Coverage : {cov} \t Error : {auto_err}')
            

            D['S'].append(S.detach().cpu().numpy())
            D['A'].append(A.detach().cpu().numpy())
            D['B'].append(B.detach().cpu().numpy())

            D["err_"].append(err_surrogate.detach().cpu().numpy() )
            D["cov_"].append(cov_surrogate.detach().cpu().numpy() )
            D["err"].append(auto_err.detach().cpu().numpy() )
            D["cov"].append(cov.detach().cpu().numpy() )
            D["loss"].append(loss.detach().cpu().numpy())

            #if(auto_err + self.C_1 * torch.sqrt(auto_err*(1-auto_err)) <=  self.eps and cov > 0.01 ):
            #    break
            #if(torch.abs(auto_err-self.C_1*self.eps)<=0.001):
            #    break
            
            epoch += 1
        
        return D 
    

    def predict(self, ds, inference_conf=None):

        self.auto_model.eval()

        g_model = self.auto_model.g_model
        t       = self.auto_model.t 
        device        = self.calib_conf['device']
        clf_inf_out = self.cur_clf.predict(ds)

        features    = clf_inf_out['logits']

        self.Y_correct = (clf_inf_out['labels'] != ds.Y).long()
        # 1 ==> incorrect prediction (mistake), 0 ==> correct prediction (no mistake)
        
        # create dataset with featuers and Y_correct labels 
        ds_ = CustomTensorDataset(X= features,Y=self.Y_correct)

        D = {"err_":[],"cov_":[], "err":[], "cov":[], "loss":[]}

        lst_cov_ = []
        lst_err_ = []
        lst_S = []
        lst_targets = []
        lst_g = []

        for inputs, targets,idx in DataLoader(ds_, batch_size=64):
            
            if isinstance(inputs,torch.Tensor):
                inputs = inputs.to(device)
            if isinstance(targets,torch.Tensor):
                targets = targets.to(device)
                
            out        = g_model.forward(inputs)
            #logits     = out['logits'].squeeze()
            logits     = safe_squeeze(out['logits'])

            g = logits
            #g          = torch.abs(logits) 

            S          = torch.clamp(torch.sigmoid(g-t),0,1)
            lst_S.append(S)
            lst_targets.append(targets)
            lst_g.append(logits)
        
        S = torch.cat(lst_S,dim=0)

        targets = torch.cat(lst_targets,dim=0)
        n = len(S)
        G = torch.cat(lst_g, dim=0)

        cov_    = torch.sum(S)/n
        
        err_    = torch.dot(S, (1-targets).float())/torch.sum(S)
        
        U       =  S>=0.5
        A       =  targets==0   # where no error is made 
        B       =  targets==1   # where error is made

        auto_err = torch.sum(torch.logical_and(U,B))/(torch.sum(U)+1)
        cov      = (torch.sum(U))/n 
        
        inf_out = clf_inf_out
        #inf_out['confidence'] = G.detach().cpu().numpy() 
        inf_out['confidence'] = S.detach().cpu().numpy() 

        return inf_out 

