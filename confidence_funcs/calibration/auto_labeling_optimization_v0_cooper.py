import sys 
sys.path.append('../')
sys.path.append('../../')

import torch
import torch.nn.functional as F  
from .abstract_calibrator import AbstractCalibrator
from collections import OrderedDict
import cooper

from src.optimizers import optim_factory
#from src.classifiers.torch.models.mlp_head import * 
from src.classifiers.torch.models.dynamic_mlp import * 
from src.data_layer.dataset_utils import * 
from src.utils.torch_shenanigans import * 

def get_S(t_model, out, y_hat,idx, calib_conf,num_classes):
    
    alpha_1 = 1.0 

    logits     = safe_squeeze(out['logits'])
    confidence = safe_squeeze(out['probs']) 
    g = confidence 
    #g = logits 

    use_t = True 

    if(calib_conf.class_wise=="joint"):
        t          = safe_squeeze(list(t_model.parameters())[0])    
        g          = logits
        S          = torch.sigmoid( (g- t)*alpha_1 )
        
    elif(calib_conf.class_wise=="independent"):
        #g          = torch.abs(logits) 
        #t           = safe_squeeze(list(self.t.parameters())[0])
        
        y_hat_1_hot = F.one_hot(y_hat,num_classes= num_classes)
        g_y         = torch.sum(g*y_hat_1_hot, dim=1) 
        if(use_t):
            t_y         =  t_model(y_hat_1_hot.float())
            t_y         =  safe_squeeze(t_y)

            S           = torch.sigmoid( (g_y- t_y)*alpha_1 )
            #S  = F.relu((g_y- t_y)*self.alpha_1)
        else:
            S           = g_y #torch.sigmoid( (g_y)*self.alpha_1 )
        #S = confidence 
        g = g_y 

    elif(calib_conf.class_wise == "joint_g_independent_t"):

        y_hat_1_hot = F.one_hot(y_hat,num_classes= num_classes)

        if(use_t):
            t_y         = t_model(y_hat_1_hot.float())
            t_y         =  safe_squeeze(t_y)
            S          = torch.sigmoid( (g- t_y)*alpha_1 )
        else:
            S           = torch.sigmoid( (g)*alpha_1 )

    return S, g 

class ConstrainedAutoLblOpt(cooper.ConstrainedMinimizationProblem):
    def __init__(self, calib_conf, num_classes):
        self.num_classes = num_classes 
        self.calib_conf = calib_conf
         
        self.criterion = torch.nn.CrossEntropyLoss()
        super().__init__(is_constrained=True)

    def closure(self, inputs, auto_model, y_hat, Y_correct, idx,eps):
        
        g_model    = auto_model.g_model 
        t_model    = auto_model.t 
        out        = g_model.forward(inputs) 
        #print(input, out, g_model)   
        

        logits     = safe_squeeze(out['logits'])
        confidence = safe_squeeze(out['probs'])

        S,g =  get_S(t_model , out,  y_hat,idx, self.calib_conf, self.num_classes)
        targets    = Y_correct[idx]

        #print(S.shape)
        n = len(S)

        cov_surrogate    = torch.sum(S)/n 
        loss = -cov_surrogate 

        if(cov_surrogate > 1e-4):
            err_surrogate    = torch.dot(S, targets.float())/torch.sum(S)
        else:
            err_surrogate =   (torch.dot(S, targets.float()) + 1e-2)/(torch.sum(S)+1e-2)
        
        err_surrogate = torch.clamp(err_surrogate, 1e-4, 1-1e-5)

        #u = self.C_1 * torch.sqrt(err_surrogate * (1-err_surrogate) ) 

        #loss = -cov_surrogate*l1 + l2*(err_surrogate - self.eps) # *torch.abs((err_surrogate + u -  self.eps ))
        
        ineq_defect = err_surrogate-eps

        print(loss.item(), ineq_defect.item())
        # We want each row of W to have norm less than or equal to 1
        # g(W) <= 1  ---> g(W) - 1 <= 0
        
        return cooper.CMPState(loss=loss, ineq_defect=ineq_defect, eq_defect=None)

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

    def forward(self, x):
        pass 


class MixedAutoLabelingModel(nn.Module):
    def __init__(self, model_conf):
        super(MixedAutoLabelingModel,self).__init__()
        
        k  = model_conf['num_classes']
        model_conf['output_dimension'] =  1
        self.g_model =  DynamicMLP(model_conf)
        self.t       =  torch.nn.Linear(k,1,bias=False)
        #self.alpha_1 = 1.0

class AutoLabelingOptimization_V0_Cooper(AbstractCalibrator):

    def __init__(self,clf, calib_conf,logger):

        self.calib_conf = calib_conf
        # the classifier model for which we are learning g and t
        self.cur_clf = clf  
        self.logger = logger 

    def init(self,ds_calib, clf_inf_out=None):

        calib_conf    = self.calib_conf 
        cur_clf       = self.cur_clf
        self.logger.info(calib_conf)

        #train_conf    = self.calib_conf.training_conf
        train_conf_g    = self.calib_conf.training_conf_g
        train_conf_t    = self.calib_conf.training_conf_t
        
        auto_lbl_conf = calib_conf.auto_lbl_conf

        if(clf_inf_out is None):
            clf_inf_out = cur_clf.predict(ds_calib)

        features    = clf_inf_out[self.calib_conf['features_key']]

        self.eps    = auto_lbl_conf.auto_label_err_threshold


        self.Y_correct = (clf_inf_out['labels'] != ds_calib.Y).long()
        # 1 ==> incorrect prediction (mistake), 0 ==> correct prediction (no mistake)

        
        # create dataset with featuers and Y_correct labels 
        self.ds_calib_train = CustomTensorDataset(X= features,Y=clf_inf_out['labels'])

        self.C_1 = self.calib_conf['auto_lbl_conf']['C_1']

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


    def fit(self,calib_ds,calib_ds_inf_out=None):
        
        self.init(calib_ds,calib_ds_inf_out)
        logger = self.logger 

        calib_conf    = self.calib_conf
        device        = calib_conf['device']
        print(device)
        #train_conf    = self.calib_conf.training_conf

        epochs = self.calib_conf.training_conf_g['max_epochs']

        logger.debug(self.calib_conf.training_conf_g)
        logger.debug(self.calib_conf.training_conf_t)

        self.auto_model.train()
        self.auto_model = self.auto_model.to(device)
        
        batch_size = calib_conf['training_conf_g']['batch_size']

        self.Y_correct = self.Y_correct.to(device)

        l1 = self.calib_conf['l1']
        l2 = self.calib_conf['l2']
        l3 = self.calib_conf['l3']
        logger.debug(f"{l1}, {l2}, {l3}")
        #l4 = self.calib_conf['l4']
        #mse_loss = torch.nn.MSELoss() 

        cmp = ConstrainedAutoLblOpt(calib_conf= self.calib_conf, num_classes=self.num_classes)
        formulation = cooper.LagrangianFormulation(cmp)

        

        # primal_optimizer = cooper.optim.ExtraSGD(model.parameters(), lr=1e-3, momentum=0.9)
        # dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraSGD, lr=5e-3)

        primal_optimizer = torch.optim.Adam(self.auto_model.parameters(), lr=5e-3, weight_decay=0.1)
        dual_optimizer = cooper.optim.partial_optimizer(torch.optim.Adam, lr=1e-3,weight_decay=0.1)

        coop = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

        ter_num = 0

        state_history = OrderedDict()

        for iter_num in range(1000):
            for inputs, y_hat, idx in DataLoader( self.ds_calib_train, batch_size=batch_size, shuffle=True):
                
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(y_hat,torch.Tensor):
                    y_hat = y_hat.to(device)

                coop.zero_grad()
                lagrangian = formulation.composite_objective(
                    cmp.closure, inputs, self.auto_model, y_hat, self.Y_correct, idx, self.eps
                )
                formulation.custom_backward(lagrangian)
                coop.step(cmp.closure, inputs, self.auto_model, y_hat, self.Y_correct, idx, self.eps)

                if iter_num % 5 == 0:
                    state_history[iter_num] = {
                        "cmp": cmp.state
                    }

                iter_num += 1

        
        return 0
    

    def predict(self, ds, inference_conf=None):


        self.g_model.eval()
        self.t.eval()

        device        = self.calib_conf['device']

        clf_inf_out = self.cur_clf.predict(ds)

        features    = clf_inf_out[self.calib_conf['features_key']]

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
        alpha_1 = self.calib_conf['alpha_1']
        calib_conf = self.calib_conf 

        for inputs, y_hat,idx in DataLoader(ds_, batch_size=512):
            
            inputs = inputs.to(device)
            y_hat = y_hat.to(device)
            
            out        = self.g_model.forward(inputs)
                
            #print(out['logits'].shape, targets.shape)

            logits     = safe_squeeze(out['logits'])
            confidence = safe_squeeze(out['probs'])
            S,g = get_S(self.t , out,  y_hat,idx, self.calib_conf, self.num_classes)
            #S,g = self.get_S(out,y_hat,idx)


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

