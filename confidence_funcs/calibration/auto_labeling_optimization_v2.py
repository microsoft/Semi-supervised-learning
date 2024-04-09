import sys 
sys.path.append('../')
sys.path.append('../../')

import torch
from .abstract_calibrator import AbstractCalibrator

from src.optimizers import optim_factory
#from src.classifiers.torch.models.mlp_head import * 
from src.classifiers.torch.models.dynamic_mlp import * 
from src.data_layer.dataset_utils import * 

from src.utils.torch_shenanigans import * 



class AutoLabelingModel(nn.Module):
    def __init__(self, model_conf):
        super(AutoLabelingModel,self).__init__()
        
        k  = model_conf['num_classes']

        model_conf['output_dimension'] =  k 
        
        self.g_model =  DynamicMLP(model_conf)
        #self.t       =  torch.nn.Parameter(torch.tensor([0.95]))
        
        #self.t       = torch.nn.Parameter(torch.rand(k))

    

class AutoLabelingOptimization_V2(AbstractCalibrator):

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
        #train_conf_t    = self.calib_conf.training_conf_t
        
        auto_lbl_conf = calib_conf.auto_lbl_conf

        if(clf_inf_out is None):
            clf_inf_out = cur_clf.predict(ds_calib)

        features    = clf_inf_out[self.calib_conf['features_key']]

        self.eps    = auto_lbl_conf.auto_label_err_threshold

        self.C_1    = auto_lbl_conf.C_1


        self.Y_correct = (clf_inf_out['labels'] != ds_calib.Y).long()
        # 1 ==> incorrect prediction (mistake), 0 ==> correct prediction (no mistake)

        
        # create dataset with featuers and Y_correct labels 
        #self.ds_calib_train = CustomTensorDataset(X= features,Y=self.Y_correct)

        

        self.ds_calib_train = CustomTensorDataset(X= features,Y=clf_inf_out['labels'])

        if("model_conf" not in calib_conf):
            g_model_conf = {} 
        else:
            g_model_conf = calib_conf.model_conf
        
        if("input_dimension" not in g_model_conf or g_model_conf['input_dimension']==-1):
            g_model_conf["input_dimension"] = features.shape[1]

        if("num_classes" not in g_model_conf):
            if("num_classes" in calib_conf):
                g_model_conf["num_classes"] = calib_conf["num_classes"]
            else:
                g_model_conf["num_classes"] = features.shape[1]
        
        self.num_classes    = g_model_conf["num_classes"] 

        self.auto_model     = AutoLabelingModel(g_model_conf)

        self.logger.info(self.auto_model.g_model)
        
        self.optimizer_g, self.lr_scheduler_g = optim_factory.get_optimizer(self.auto_model,train_conf_g)
        #self.optimizer_t, self.lr_scheduler_t = get_optimizer(self.auto_model,train_conf_t)


    def fit(self,calib_ds,calib_ds_inf_out=None):
        
        self.init(calib_ds,calib_ds_inf_out)
        logger = self.logger 

        calib_conf    = self.calib_conf
        device        = calib_conf['device']
        #train_conf    = self.calib_conf.training_conf

        epochs = self.calib_conf.training_conf_g['max_epochs']

        logger.debug(self.calib_conf.training_conf_g)
        #logger.debug(self.calib_conf.training_conf_t)

        self.auto_model.train()
        self.auto_model = self.auto_model.to(device)

        g_model = self.auto_model.g_model
        #t       = self.auto_model.t 

        D = {"err_":[],"cov_":[], "err":[], "cov":[], "loss":[], "S":[],'A':[],'B':[]}
        
        logger.debug('Using optimizer for g: {}'.format(type(self.optimizer_g)))
        #logger.debug('Using optimizer for t: {}'.format(type(self.optimizer_t)))

        batch_size = calib_conf['training_conf_g']['batch_size']

        xent = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            
            
            
            lst_cov_ = []
            lst_err_ = []
            lst_S = []
            lst_targets = []
            lst_g = []
            l1 = self.calib_conf['l1']
            l2 = self.calib_conf['l2']
            l3 = self.calib_conf['l3']
            l4 = self.calib_conf['l4']
            alpha_1 = self.calib_conf['alpha_1']

            self.Y_correct = self.Y_correct.to(device)
            total_loss = 0.0 #torch.tensor(0.0)

            for inputs, y_hat,idx in DataLoader( self.ds_calib_train, batch_size=batch_size , shuffle=True):
                self.optimizer_g.zero_grad()
                
                #inputs.requires_grad_(True)
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(y_hat,torch.Tensor):
                    y_hat = y_hat.to(device)

                out        = g_model.forward(inputs)
                logits     = safe_squeeze(out['logits'])
                confidence = safe_squeeze(out['probs'])

                g          = confidence
                targets    = self.Y_correct[idx]
                #g          = logits        

                #S          = torch.clamp(torch.sigmoid(g-t),0,1)
                y_hat_1_hot = F.one_hot(y_hat,num_classes= self.num_classes)

                g_y_hat     = torch.sum(g*y_hat_1_hot,dim=1)
                #t_y_hat     = torch.sum(t*y_hat_1_hot, dim=1 )

                l_y_hat     = torch.sum(logits*y_hat_1_hot,dim=1)

                n = len(g_y_hat)

                
                
                S = g_y_hat
                cov_surrogate    = torch.sum(S)/n 
                #err_surrogate    = (torch.dot(S, targets.float()) + 1e-2)/(torch.sum(S) + 1.0)
                if(cov_surrogate <=1e-5):
                    err_surrogate    = torch.dot(S, targets.float())/n 
                    # to avoid nan issue
                else:
                    err_surrogate    = torch.clamp( torch.dot(S, targets.float())/torch.sum(S), 1e-4, 1-1e-4)
                
                #err_surrogate = xent(logits,y_hat)/torch.sum(S)

                #err_surrogate    = torch.dot(S, targets.float()) /n

                #logger.debug(f'cov_surrogate :  {cov_surrogate}    err_surrogate {err_surrogate}')

                u = self.C_1 * torch.sqrt(err_surrogate * (1-err_surrogate) + 1e-4) 

                loss3  = torch.mean((S*targets)**2)  # encourage lower scores for incorrect predictions
                
                loss4 = xent(logits,y_hat)           # encourage higher scores corresponding to y_hat ( encourage it to be consistent with h)

                loss1 = -cov_surrogate 
                loss2 = torch.abs((err_surrogate + u -  self.eps ))

                loss = l1*loss1 + l2*loss2 +  l3*loss3 + l4*loss4   

                #loss = -cov_surrogate*l1 + l2*torch.abs((err_surrogate -  self.eps*0.5 ))
                #loss = -cov_surrogate*l1 + l2*torch.square((err_surrogate -  self.eps*0.5 ))
                #loss = -cov_surrogate*l1 + l2*torch.square((err_surrogate -  self.eps*0.5 ))

                if(calib_conf['regularize']):
                    reg   = torch.mean((torch.mul(logits, 1-F.one_hot(targets,num_classes=self.num_classes)))**2)

                    loss+= reg
                     
                #G = torch.cat(lst_g,dim=0)
            
                #targets2 = -(2*targets -1)  # -1 for incorrect, 1 for correct
            
                #loss3  = torch.mean((g_y_hat -targets2)**2)

                #targets3 = 1-targets
                #loss3  = torch.mean((S -targets3)**2)
                #loss3   = -torch.log(torch.mean((S*targets)))
                
                loss.backward() 

                self.optimizer_g.step()

                
                total_loss += loss.item() 

            


                #logger.debug(f'Loss: {loss.item()}')

                S = S.detach() 
                g = g.detach()

                lst_S.append(S)
                lst_g.append(g)

                lst_targets.append(targets.detach())

            self.lr_scheduler_g.step()
            #self.lr_scheduler_t.step()

            S       = torch.cat(lst_S,dim=0)
            targets = torch.cat(lst_targets,dim=0)
            n       = len(S)
            #G = torch.cat(lst_g,dim=0)
            
            logger.debug(f'Epoch: {epoch} Loss :{total_loss}')

            cov_surrogate    = torch.sum(S)/n 

            #targets = self.Y_correct[idx]
            
            err_surrogate    = torch.dot(S, targets.float())/torch.sum(S)

            # for logging and checking progress
            U       =  S>=0.5
            A       =  targets==0   # where no error is made 
            B       =  targets==1   # where error is made
            
            auto_err = torch.sum(torch.logical_and(U,B))/(torch.sum(U)+1)
            cov      = (torch.sum(U))/n 
            
            logger.debug(f"Epoch: {epoch} Surrogate Coverage : {cov_surrogate.item()} Surrogate Error: {err_surrogate.item()} ")
            
            logger.debug(f'Epoch: {epoch} At t = 0.5, {epoch} Coverage : {cov} \t Error : {auto_err}')
            

            #D['S'].append(S.detach().cpu().numpy())
            #D['A'].append(A.detach().cpu().numpy())
            #D['B'].append(B.detach().cpu().numpy())

            #D["err_"].append(err_surrogate.detach().cpu().numpy() )
            #D["cov_"].append(cov_surrogate.detach().cpu().numpy() )
            #D["err"].append(auto_err.detach().cpu().numpy() )
            #D["cov"].append(cov.detach().cpu().numpy() )
            #D["loss"].append(loss.detach().cpu().numpy())

            #if(auto_err + self.C_1 * torch.sqrt(auto_err*(1-auto_err)) <=  self.eps and cov > 0.01 ):
            #    break
            #if(torch.abs(auto_err-self.C_1*self.eps)<=0.001):
            #    break
            
            epoch += 1
        
        return D 
    

    def predict(self, ds, inference_conf=None):

        self.auto_model.eval()

        g_model = self.auto_model.g_model
        
        g_model.eval()
        
        device        = self.calib_conf['device']
        clf_inf_out = self.cur_clf.predict(ds)

        features    = clf_inf_out[self.calib_conf['features_key']]

        Y_correct = (clf_inf_out['labels'] != ds.Y).long()
        # 1 ==> incorrect prediction (mistake), 0 ==> correct prediction (no mistake)
        
        # create dataset with featuers and Y_correct labels 
        ds_ = CustomTensorDataset(X= features,Y=clf_inf_out['labels'])

        D = {"err_":[],"cov_":[], "err":[], "cov":[], "loss":[]}

        lst_cov_ = []
        lst_err_ = []
        lst_S = []
        lst_targets = []
        lst_g = []

        Y_correct = Y_correct.to(device)
        alpha_1 = self.calib_conf['alpha_1']
        lst_logits = []
        for inputs, y_hat,idx in DataLoader(ds_, batch_size=2048):
            
            if isinstance(inputs,torch.Tensor):
                inputs = inputs.to(device)
            if isinstance(y_hat,torch.Tensor):
                y_hat = y_hat.to(device)
                
            out        = g_model.forward(inputs)

            logits     = safe_squeeze(out['logits'])
            confidence = safe_squeeze(out['probs'])

            #g = logits
            g = confidence 
            #g          = torch.abs(logits) 

            #S          = torch.clamp(torch.sigmoid(g-t),0,1)

            y_hat_1_hot = F.one_hot(y_hat,self.num_classes)

            g_y_hat     = torch.sum(g*y_hat_1_hot,dim=1)
            l_y_hat     = torch.sum(logits*y_hat_1_hot,dim=1)
            #g_y_hat     = torch.sum(confidence*y_hat_1_hot,dim=1)
            lst_logits.append(logits.detach().cpu().numpy())


            #S = torch.sigmoid(g_y_hat-t_y_hat)
            #S = torch.sigmoid( alpha_1* (g_y_hat-t_y_hat))
            

            S = g_y_hat

            lst_S.append(S.detach())
            #lst_targets.append(targets.detach())
            lst_g.append(g_y_hat.detach())
        
        S = torch.cat(lst_S,dim=0)

        targets =  Y_correct #torch.cat(lst_targets,dim=0)
        n = len(S)
        G = torch.cat(lst_g, dim=0)

        cov_    = torch.sum(S)/n
        
        err_    = torch.dot(S, (1-targets).float())/torch.sum(S)

        #print(cov_, err_ )
        
        U       =  S>=0.5
        A       =  targets==0   # where no error is made 
        B       =  targets==1   # where error is made

        auto_err = torch.sum(torch.logical_and(U,B))/(torch.sum(U)+1)
        cov      = (torch.sum(U))/n 
        
        inf_out = clf_inf_out

        #inf_out['logits']   = G.detach().cpu().numpy() 
        inf_out['logits']   = torch.Tensor(np.vstack(lst_logits))
        #inf_out['confidence'] = S.detach().cpu().numpy() 
        #inf_out['confidence'] = G.detach().cpu().numpy() 
        inf_out['confidence'] = S.detach().cpu().numpy() 

        #inf_out['confidence'] = S.detach().cpu().numpy() 

        return inf_out 

