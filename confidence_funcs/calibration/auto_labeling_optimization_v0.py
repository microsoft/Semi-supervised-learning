import sys 
sys.path.append('../')
sys.path.append('../../')

import torch
import torch.nn.functional as F  
from .abstract_calibrator import AbstractCalibrator

from src.optimizers import optim_factory
#from src.classifiers.torch.models.mlp_head import * 
from src.classifiers.torch.models.dynamic_mlp import * 
from src.data_layer.dataset_utils import * 
from src.utils.torch_shenanigans import * 

from src.core.threshold_estimation import * 

import copy 
class JointAutoLabelingModel(nn.Module):
    def __init__(self, model_conf):
        super(JointAutoLabelingModel,self).__init__()
        
        k  = model_conf['num_classes']
        model_conf['output_dimension'] =  1
        self.g_model =  DynamicMLP(model_conf)
        self.t       =  torch.nn.Linear(1,1,bias=False)
        #self.alpha_1 = 1.0

  
class ClasswiseAutoLabelingModel(nn.Module):
    def __init__(self, model_conf):
        super(ClasswiseAutoLabelingModel,self).__init__()
        
        k  = model_conf['num_classes']
        model_conf['output_dimension'] =  k
        self.g_model =  DynamicMLP(model_conf)
        self.t       =  torch.nn.Linear(k,1,bias=False)
        #self.alpha_1 = 1.0

class MixedAutoLabelingModel(nn.Module):
    def __init__(self, model_conf):
        super(MixedAutoLabelingModel,self).__init__()
        
        k  = model_conf['num_classes']
        model_conf['output_dimension'] =  1
        self.g_model =  DynamicMLP(model_conf)
        self.t       =  torch.nn.Linear(k,1,bias=False)
        #self.alpha_1 = 1.0

class AutoLabelingOptimization_V0(AbstractCalibrator):

    def __init__(self,clf, calib_conf,logger):

        self.calib_conf = calib_conf
        # the classifier model for which we are learning g and t
        self.cur_clf = clf  
        self.logger = logger 
        self.g_model = None 

    def init(self,ds_calib, clf_inf_out=None, use_prev_model=False ):

        calib_conf    = self.calib_conf 
        cur_clf       = self.cur_clf
        #print(calib_conf)
        
        #print(clf_inf_out)

        self.lst_classes = [i for i in range(calib_conf.num_classes)]

        self.logger.info(calib_conf)

        #train_conf    = self.calib_conf.training_conf
        train_conf_g    = self.calib_conf.training_conf_g
        train_conf_t    = self.calib_conf.training_conf_t
        
        self.auto_lbl_conf = calib_conf.auto_lbl_conf

        if(clf_inf_out is None):
            clf_inf_out = cur_clf.predict(ds_calib)
        
        fkey = self.calib_conf['features_key'] 


        if(fkey == 'concat'):
            features    = torch.hstack([ clf_inf_out['logits'], clf_inf_out['pre_logits'] ])
        
        else:
            features    = clf_inf_out[self.calib_conf['features_key']]
        
        self.Y_correct = (clf_inf_out['labels'] != ds_calib.Y).long()
        # 1 ==> incorrect prediction (mistake), 0 ==> correct prediction (no mistake)
        
        # create dataset with featuers and Y_correct labels 
        self.ds_calib_train = CustomTensorDataset(X= features,Y=clf_inf_out['labels'])

        self.logger.debug(f"**** len(features) {len(features)}")
        
        self.logger.debug(f'Features Shape : {features.shape}')

        self.eps    = self.auto_lbl_conf.auto_label_err_threshold


        self.C_1 = self.calib_conf['auto_lbl_conf']['C_1']

        if(not use_prev_model):
            print('Here **********************************************')

            if("model_conf" not in calib_conf):
                g_model_conf = {} 
            else:
                g_model_conf = calib_conf.model_conf
            
            if("input_dimension" not in g_model_conf or g_model_conf["input_dimension"]==-1):
                g_model_conf["input_dimension"] = features.shape[1]

            if("num_classes" not in g_model_conf):
                g_model_conf["num_classes"] = calib_conf.num_classes #features.shape[1]
            
            k  = g_model_conf['num_classes']

            if(calib_conf.class_wise=="joint"):
                self.auto_model     = JointAutoLabelingModel(g_model_conf)

            elif(calib_conf.class_wise=="independent"):
                self.auto_model     = ClasswiseAutoLabelingModel(g_model_conf)
            elif(calib_conf.class_wise == "joint_g_independent_t"):
                self.auto_model     = MixedAutoLabelingModel(g_model_conf)
            
            self.logger.info(f'Using auto-labeling model : {self.auto_model}') 

            self.g_model = self.auto_model.g_model
            self.t       = self.auto_model.t 

            self.alpha_1 = calib_conf['alpha_1'] # self.auto_model.alpha_1 

            self.num_classes = g_model_conf['num_classes']
            
            self.logger.info(f"g model {self.g_model}")
            
            self.optimizer_g, self.lr_scheduler_g = optim_factory.get_optimizer(self.g_model,train_conf_g)
            self.optimizer_t, self.lr_scheduler_t = optim_factory.get_optimizer(self.t, train_conf_t)
        

    def get_S(self, out, y_hat,idx):
        logits     = safe_squeeze(out['logits'])
        confidence = safe_squeeze(out['probs']) 
        #g = confidence 
        #logits = logits
        #print(torch.norm(logits).item())

        #g = torch.abs(logits)
        #print()
        #logits = logits/ torch.norm(logits)
        

        g = confidence 
        #g = logits 

        use_t = True

        if(self.calib_conf.class_wise=="joint"):
            t          = safe_squeeze(list(self.t.parameters())[0])    
            g          = logits
            S          = torch.sigmoid( (g- t)*self.alpha_1 )
            
        elif(self.calib_conf.class_wise=="independent"):
            #g          = torch.abs(logits) 
            #t           = safe_squeeze(list(self.t.parameters())[0])
            
            y_hat_1_hot = F.one_hot(y_hat,num_classes= self.num_classes)
            g_y         = torch.sum(g*y_hat_1_hot, dim=1) 
            if(use_t):
                t_y         = self.t(y_hat_1_hot.float())
                t_y         =  safe_squeeze(t_y)

                S           = torch.sigmoid( (g_y- t_y)*self.alpha_1 )
                #S  = F.relu((g_y- t_y)*self.alpha_1)
            else:
                S           = torch.sigmoid( (g_y)*self.alpha_1 )
            #S = confidence 
            g = g_y 

        elif(self.calib_conf.class_wise == "joint_g_independent_t"):

            y_hat_1_hot = F.one_hot(y_hat,num_classes= self.num_classes)

            if(use_t):
                t_y         = self.t(y_hat_1_hot.float())
                t_y         =  safe_squeeze(t_y)
                S          = torch.sigmoid( (g- t_y)*self.alpha_1 )
            else:
                S           = torch.sigmoid( (g)*self.alpha_1 )

        return S, g 


    def fit(self,calib_ds,calib_ds_inf_out=None,ds_val_nc=None):
        
        
        use_prev_model = self.calib_conf['use_prev_model']  and self.g_model is not None

        
        print(use_prev_model)

        self.init(calib_ds,calib_ds_inf_out, use_prev_model=use_prev_model)
        logger = self.logger 

        calib_conf    = self.calib_conf
        device        = calib_conf['device']
        self.ds_val_nc = ds_val_nc
        #train_conf    = self.calib_conf.training_conf

        epochs = self.calib_conf.training_conf_g['max_epochs']

        logger.debug(self.calib_conf.training_conf_g)
        logger.debug(self.calib_conf.training_conf_t)

        #self.auto_model.train()
        #self.auto_model = self.auto_model.to(device)

        self.g_model.train()
        self.t.train()

        self.g_model = self.g_model.to(device)
        self.t       = self.t.to(device)
        
        
        self.Y_correct = self.Y_correct.to(device)

        logger.info(f'Length of calibration dataset: {len(calib_ds)}')
        
        logger.debug('Using optimizer for g: {}'.format(type(self.optimizer_g)))
        logger.debug('Using optimizer for t: {}'.format(type(self.optimizer_t)))
        
        batch_size = calib_conf['training_conf_g']['batch_size']
        
        alpha_1 = self.calib_conf['alpha_1']

        l1 = self.calib_conf['l1']
        l2 = self.calib_conf['l2']
        l3 = self.calib_conf['l3']
        logger.debug(f"{l1}, {l2}, {l3}")
        #l4 = self.calib_conf['l4']
        mse_loss = torch.nn.MSELoss() 

        max_cov_val_nc = 0
        best_model_so_far = [copy.deepcopy(self.g_model), copy.deepcopy(self.t)]
        for epoch in range(epochs):
            
            self.optimizer_g.zero_grad()
            self.optimizer_t.zero_grad()
            
            lst_cov_ = []
            lst_err_ = []
            lst_S = []
            lst_targets = []
            lst_g = []
            total_loss = 0.0
            lr_g = self.optimizer_g.param_groups[0]['lr']
            logger.debug(f"Epoch: {epoch} Learning Rate g : {lr_g}")
            
            #print(len(self.ds_calib_train))

            for inputs, y_hat, idx in DataLoader( self.ds_calib_train, batch_size=batch_size, shuffle=True):
                
                #inputs.requires_grad_(True)
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(y_hat,torch.Tensor):
                    y_hat = y_hat.to(device)

                #S = self.auto_model.forward(inputs)

                out        = self.g_model.forward(inputs)
                
                #print(out['logits'].shape, targets.shape)

                logits     = safe_squeeze(out['logits'])
                confidence = safe_squeeze(out['probs'])

                S,g = self.get_S(out,y_hat,idx)
              

                targets    = self.Y_correct[idx]

                #print(S.shape)
                n = len(S)

                cov_surrogate    = torch.sum(S)/n 
                if(cov_surrogate > 1e-4):
                    err_surrogate    = torch.dot(S, targets.float())/torch.sum(S)
                else:
                    err_surrogate =   (torch.dot(S, targets.float()) + 1e-2)/(torch.sum(S)+1e-2)
                
                err_surrogate = torch.clamp(err_surrogate, 1e-4, 1-1e-5)

                #u = self.C_1 * torch.sqrt(err_surrogate * (1-err_surrogate) ) 

                loss = -cov_surrogate + l2*(err_surrogate) # - self.eps) # *torch.abs((err_surrogate + u -  self.eps ))
                
                #loss3  = torch.mean(S*targets) 
                #loss3  = loss3 -torch.mean(S* (1-targets)) 
                #loss3 = torch.mean( (S - (1-targets)))
                #print(S)
                #print(1-targets)
                loss3 = mse_loss(S, 1.0-targets)

                loss += l3 * loss3

                if(calib_conf['regularize']):
                    reg   = torch.mean((torch.mul(logits, 1-F.one_hot(targets,num_classes=self.num_classes)))**2)

                    loss+= reg

                
                loss.backward() 

                self.optimizer_g.step()

                self.optimizer_t.step()
                
                lst_S.append(S.detach())
                #lst_g.append(g.detach())

                lst_targets.append(targets.detach())

                total_loss += n * loss.item() 

            S       = torch.cat(lst_S,dim=0)
            targets = torch.cat(lst_targets,dim=0)
            N       = len(S)

            cov_surrogate    = torch.sum(S)/N
            
            err_surrogate    = torch.dot(S, targets.float())/torch.sum(S)
            
            logger.debug(f'Epoch: {epoch} Loss :{total_loss/N}')


            # for logging and checking progress
            U       =  S>=0.5
            A       =  targets==0   # where no error is made 
            B       =  targets==1   # where error is made
            
            auto_err = torch.sum(torch.logical_and(U,B))/(torch.sum(U)+1)
            cov      = (torch.sum(U))/N
            
            logger.debug(f"Epoch: {epoch} Surrogate Coverage : {cov_surrogate.item()} Surrogate Error: {err_surrogate.item()} ")
            
            logger.debug(f'Epoch: {epoch} At t = 0.5, {epoch} Coverage : {cov} \t Error : {auto_err}')

            logger.debug(f"Epoch: {epoch} Model Norm  {model_norm(self.g_model)}")

            if self.ds_val_nc:
                if self.num_classes > 20 and epoch % 10 == 0 or self.num_classes <= 20 and epoch % 10 == 0:
                    cov_val_nc = self.eval(self.ds_val_nc)
                    logger.debug(f"Epoch : {epoch}  Actual Coverage on NC VAL : {cov_val_nc}")
                    if(cov_val_nc>max_cov_val_nc):
                        max_cov_val_nc = cov_val_nc
                        best_model_so_far = [copy.deepcopy(self.g_model), copy.deepcopy(self.t)]


            #D['S'].append(S.detach().cpu().numpy())
            #D['A'].append(A.detach().cpu().numpy())
            #D['B'].append(B.detach().cpu().numpy())

            #D["err_"].append(err_surrogate.detach().cpu().numpy() )
            #D["cov_"].append(cov_surrogate.detach().cpu().numpy() )
            #D["err"].append(auto_err.detach().cpu().numpy() )
            #D["cov"].append(cov.detach().cpu().numpy() )
            #D["loss"].append(loss.detach().cpu().numpy())


            #if(torch.abs(auto_err-self.eps)<=0.01):
            #    break
            
            epoch += 1
            self.lr_scheduler_g.step()
            self.lr_scheduler_t.step()
        
        self.g_model, self.t = best_model_so_far[0],best_model_so_far[1]


        return 0
    

    def predict(self, ds, inference_conf=None):
        
        self.g_model.eval()
        self.t.eval()

        device        = self.calib_conf['device']

        clf_inf_out = self.cur_clf.predict(ds)

        #features    = clf_inf_out[self.calib_conf['features_key']]
        
        fkey = self.calib_conf['features_key'] 

        if(fkey == 'concat'):
            features    = torch.hstack([ clf_inf_out['logits'], clf_inf_out['pre_logits'] ])
        
        else:
            features    = clf_inf_out[self.calib_conf['features_key']]

        Y_correct = (clf_inf_out['labels'] != ds.Y).long()
        # 1 ==> incorrect prediction (mistake), 0 ==> correct prediction (no mistake)
        
        # create dataset with featuers and Y_correct labels 
        ds_ = CustomTensorDataset(X= features,Y=Y_correct)

        D = {"err_":[],"cov_":[], "err":[], "cov":[], "loss":[]}

        lst_cov_ = []
        lst_err_ = []
        lst_S = []
        lst_targets = []
        lst_g = []
        alpha_1 = self.calib_conf['alpha_1']
        calib_conf = self.calib_conf 

        for inputs, y_hat,idx in DataLoader(ds_, batch_size=512):
            
            inputs = inputs.to(device)
            y_hat = y_hat.to(device)
            
            out        = self.g_model.forward(inputs)
                
            #print(out['logits'].shape, targets.shape)

            logits     = safe_squeeze(out['logits'])
            confidence = safe_squeeze(out['probs'])

            S,g = self.get_S(out,y_hat,idx)


            lst_S.append(S)
            #lst_targets.append(targets)
            lst_g.append(g)
        
        S = torch.cat(lst_S,dim=0)

        G = torch.cat(lst_g, dim=0)
        
        inf_out = clf_inf_out
        inf_out['confidence'] = G.detach().cpu().numpy() #S.detach().cpu().numpy() 
        #inf_out['confidence']  =  S.detach().cpu().numpy() 
        #S.detach().numpy(), U.detach().numpy(), auto_err, cov, A.detach().numpy(), B.detach().numpy()

        return inf_out 
    
    def eval(self, ds_val_nc):
    
        inf_out = self.predict(ds_val_nc)
        
        val_idcs = [i for i in range(len(ds_val_nc))]

        lst_t_y, val_idcs_to_rm, val_err, cov  = determine_threshold(self.lst_classes,
                                                                     inf_out,self.auto_lbl_conf, 
                                                                     ds_val_nc, val_idcs, 
                                                                     self.logger, err_threshold=self.eps)
        return cov
        

