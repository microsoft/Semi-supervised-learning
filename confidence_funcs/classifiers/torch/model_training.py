import sys 
import os
import datetime
sys.path.append('../')
sys.path.append('../../')

import numpy as np
import torch
from collections import namedtuple
from torch import nn 
from torch import optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader
from .clf_inference import *
from sklearn.metrics import accuracy_score

from confidence_funcs.calibration.calibration_utils import *
import confidence_funcs.utils.metrics as metrics
#sys.path.append('./losses')
from confidence_funcs.classifiers.torch.losses.loss_factory import get_loss_function
from confidence_funcs.optimizers import optim_factory
#import loss_factory as loss_factory

class ModelTraining:
    
    def __init__(self,logger):
        self.logger = logger

    def init(self,model,dataset, train_params={},val_set=None):
        self.set_defaults(train_params)
        
        self.optimizer, self.lr_scheduler = optim_factory.get_optimizer(model,train_params)

        #self.init_optimizer(model,train_params) 

        train_params['num_train_pts'] = len(dataset)

        self.loss_fun = get_loss_function(train_params)

    
    def set_defaults(self, train_params):
        train_params.setdefault('loss_function','std_cross_entropy')
        
        train_params.setdefault('optimizer','sgd')
        train_params.setdefault('learning_rate',1e-2)
        train_params.setdefault('max_epochs',200)
        train_params.setdefault('batch_size',32)
        train_params.setdefault('weight_decay',1e-4)
        train_params.setdefault('momentum',0.9)
        train_params.setdefault('shuffle',True) 
        train_params.setdefault('use_lr_schedule',True)

        train_params.setdefault('nesterov',False )

        train_params.setdefault('loss_tol',1e-6)
        train_params.setdefault('device','cpu')
        train_params.setdefault('stopping_criterion','max_epochs')
        train_params.setdefault('log_val_err',False)
        train_params.setdefault('log_val_ece',False)
        train_params.setdefault('log_train_ece',False)
        
        train_params.setdefault('train_err_tol',0.001) #default less than 0.1%

        # set this to 0 to disable loss prints with batches.
        train_params.setdefault('log_batch_loss_freq',20) 
        train_params.setdefault('log_val_err_freq',5)
        train_params.setdefault('log_val_ece_freq',5)
        train_params.setdefault('verbose_logging',False)
        train_params.setdefault('log_train_ece_freq',5)
        train_params.setdefault('normalize_weights',False)
        

        # train params for mixup, mixup is not turned on by default
        train_params.setdefault('mixup_alpha', 0)

    
    def mixup_data(self, x, y, mixup_alpha, device):
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def train_one_epoch(self,model,train_data_loader,train_params,epoch_state={}):
        
        logger = self.logger 

        device = train_params['device']
        mixup_alpha = train_params['mixup_alpha']
        
        epoch_num = epoch_state['epoch_num']
        epoch_loss = 0

        num_pts = len(train_data_loader.dataset)
        log_batch_loss_freq = train_params['log_batch_loss_freq']

        model.train()
        model = model.to(device)
        
        y_hat = []
        y_true = []

        
        for batch_idx, (data, target, idx) in enumerate(train_data_loader):
            if isinstance(data,torch.Tensor):
                data = data.to(device)
            if isinstance(target,torch.Tensor):
                target = target.to(device)
            
            batch_state = {"idx": idx, "target":target}

            if mixup_alpha:
                data, target, target_b, lam = self.mixup_data(data, target, mixup_alpha, device)

            self.optimizer.zero_grad()   # set gradient to 0

            out     = model.forward(data)

            probs   = out['probs']
            logits  = out['logits']
            k = probs.shape[1]

            if(len(data)>1):
                logits = logits.squeeze()
                target = target.squeeze() 
            

            batch_state["logits"] = logits

            if mixup_alpha:
                loss = lam * self.loss_fun.forward(logits, target, idx)  + (1 - lam) * self.loss_fun.forward(logits, target_b, idx)
            else:
                loss = self.loss_fun.forward(logits, target, idx)
            loss.backward()    # compute gradient
            
            if train_params['optimizer'] == 'sam':
                self.optimizer.step(zero_grad=True)             
            else:
                self.optimizer.step()             
                
            epoch_loss += loss.item()

            if train_params['optimizer'] == 'sam':
                _logits = model.forward(data)['logits']
                if mixup_alpha:
                    (lam * self.loss_fun.forward(_logits, target, idx)  + (1 - lam) * self.loss_fun.forward(_logits, target_b, idx)).backward()
                else:
                    self.loss_fun.forward(_logits, target, idx).backward()
                self.optimizer.second_step(zero_grad=True)


            confidence, y_hat_ = torch.max(probs, 1)
            
            y_hat.extend(y_hat_.cpu().numpy())
            y_true.extend(target.cpu().numpy())

            if mixup_alpha:
                prec, correct = metrics.accuracy(probs.detach(), target.detach(), lam=lam, target_b=target_b)
            else:
                prec, correct = metrics.accuracy(probs.detach(), target.detach())
            batch_state['correct'] = correct 

            if log_batch_loss_freq >0 and batch_idx%log_batch_loss_freq == 0:
                #self.logger.debug(probs[0].max().item())
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t result: {}'.format(
                                epoch_num, batch_idx * len(data), num_pts,
                                100. * batch_idx / num_pts, loss.item(), self.loss_fun.result)) 


            if(train_params['normalize_weights']):
                model.normalize_weights()
            
            self.loss_fun.batch_closure_callback(batch_state) 


        if(train_params['use_lr_schedule']):
            self.lr_scheduler.step()
    
        training_err = 1-accuracy_score(y_hat,y_true)
        epoch_loss= epoch_loss/num_pts
        
        self.loss_fun.epoch_closure_callback(epoch_state)

        return epoch_loss, training_err

    
    def train(self,model,dataset, train_params = {} , val_set=None):
        '''
            max_epochs, loss_threshold=1e-6
        '''
        #unique_id = str(os.getpid())
        #unique_ckpt_save_path = train_params['ckpt_save_path'][:-5] + unique_id + '.' + 'ckpt'

        ckpt_save_path = train_params['ckpt_save_path'] if 'ckpt_save_path' in train_params else None 

        logger = self.logger 


        self.init(model,dataset,train_params,val_set)

        train_data_loader = DataLoader( dataset=dataset,
                                        batch_size= train_params['batch_size'], 
                                        shuffle=train_params['shuffle'],
                                        pin_memory=True,
                                        num_workers=2)

        train_loss = 0
        epoch_loss = 1e10
        epoch = 0
        
        stop_crit = train_params['stopping_criterion']
        
            
        logger.debug('Training conf : {}'.format(train_params))

        logger.debug('Using loss function : {}'.format(type(self.loss_fun)))

        logger.debug('Using stopping criterion {}'.format(stop_crit))

        inf_conf = {'device':train_params['device'],'batch_size':512,'shuffle':False}
        
        verbose = train_params['verbose_logging']

        save_best_ckpt = False 
        if(ckpt_save_path is not None):
            save_best_ckpt = True 

        #train_conf['is_calib_training'] =True 
        #train_conf['model_selection_crit'] = "ece" | "auto-labeling-cov"


        if val_set is not None:
            BestModel = namedtuple('BestModel', ['val_acc', 'model'])
            best_model = BestModel(val_acc=0, model=None)

        while(epoch< train_params['max_epochs']):
            epoch_state = {'epoch_num': epoch}
            
            lr = self.optimizer.param_groups[0]['lr']

            if(verbose):
                logger.debug('------------------------------------------------------')
                logger.debug('Training Epoch {} Begins '.format(epoch))
                logger.debug('Epoch:{} Using learning rate : {}'.format(epoch,lr))

            epoch_loss, training_err = self.train_one_epoch(model,train_data_loader,train_params,epoch_state)

            logger.debug(f'Epoch: {epoch} Training Error : {training_err:.4f} , Training Loss : {epoch_loss:.4f}')
            
            stop = False 
            val_inf_out = None 
            train_inf_out = None
            
            if(training_err <= train_params['train_err_tol']):
                stop = True 

            if(stop_crit=='max_epochs' and epoch> train_params['max_epochs']):
                stop = True 

            log_val_err_freq =     train_params['log_val_err_freq']
            log_val_ece_freq =     train_params['log_val_ece_freq']
            log_train_ece_freq = train_params['log_train_ece_freq']

            if val_set is not None and save_best_ckpt:
                val_inf_out = ClassfierInference(logger=self.logger).predict(model,val_set,inf_conf)
                val_acc = accuracy_score(val_set.Y,val_inf_out['labels'])
                curr_model = BestModel(val_acc=val_acc, model=model)
                #logger.info(f'Current Model Validation Accuracy: {curr_model.val_acc}')
                #logger.info(f'Best Model Validation Accuracy: {best_model.val_acc}')

                if curr_model.val_acc > best_model.val_acc:
                    best_model = curr_model
                    #logger.debug(f'Epoch:{epoch}, A new model with validation accuracy {val_acc} has been found!')
                    torch.save({'model_state_dict': best_model.model.state_dict()}, ckpt_save_path)
                    #logger.info(f"Saved current model checkpoint to path : {ckpt_save_path}")


            if(train_params['log_val_err'] and epoch%log_val_err_freq==0 and val_set is not None):
                val_inf_out = ClassfierInference(logger=self.logger).predict(model,val_set,inf_conf)
                val_error = 1 - accuracy_score(val_set.Y,val_inf_out['labels'])
                logger.debug('Epoch:{} Validation Error:{}'.format(epoch,val_error))
            
            if(train_params['stopping_criterion']=='val_err_threshold' and val_set is not None):
                if(val_inf_out is None):
                    val_inf_out = ClassfierInference(logger=self.logger).predict(model,val_set,inf_conf)
                
                val_error = 1 - accuracy_score(val_set.Y,val_inf_out['labels'])
                if(val_error <= train_params['val_err_threshold']):
                    stop= True 
                logger.debug('Epoch:{} Validation Error:{}'.format(epoch,val_error))

            if(train_params['log_val_ece'] and epoch%log_val_ece_freq==0):
                if(val_inf_out is None):
                    val_inf_out = ClassfierInference(logger=self.logger).predict(model,val_set,inf_conf)
                calib_out = compute_calibration(val_set.Y.numpy(),val_inf_out['labels'].numpy(), val_inf_out['confidence'].numpy(), num_bins=10)
                logger.debug('Epoch:{} Expected Calibration Error on Validation Set : {}'.format(epoch,calib_out['expected_calibration_error']))

            if(train_params['log_train_ece'] and epoch%log_train_ece_freq==0):
                if(train_inf_out is None):
                    train_inf_out = ClassfierInference(logger=self.logger).predict(model,dataset,inf_conf)
                
                if isinstance(dataset.Y, list):
                    calib_out = compute_calibration(np.array(dataset),train_inf_out['labels'].numpy(), train_inf_out['confidence'].numpy(), num_bins=10)
                else:
                    calib_out = compute_calibration(dataset.Y.numpy(),train_inf_out['labels'].numpy(), train_inf_out['confidence'].numpy(), num_bins=10)
                logger.debug('Epoch:{} Expected Calibration Error on Train Set : {}'.format(epoch,calib_out['expected_calibration_error']))


            if(stop_crit == 'loss_tol' and epoch_loss <= train_params['loss_tol']):
                stop = True 
            
            
            if(stop):    
                logger.debug('Training Stopping criterion met. ')
                logger.debug('')
                break 

            train_loss += epoch_loss
            epoch += 1
            
            if(verbose):
                logger.debug('Training Epoch {} Ends '.format(epoch))
                logger.debug('------------------------------------------------------')

        avg_train_loss = (1/(epoch+1))*train_loss
        logger.debug('Average training loss : {}'.format(avg_train_loss))

        if(save_best_ckpt):
            logger.info('Loading best model from path: {}'.format(ckpt_save_path))
            model.load_state_dict(torch.load(ckpt_save_path)['model_state_dict'])

        if val_set is not None:
            logger.info('Loading best model from path: {}'.format(ckpt_save_path))
            model.load_state_dict(torch.load(ckpt_save_path)['model_state_dict'])
            test_val_inf_out = ClassfierInference(logger=self.logger).predict(model,val_set,inf_conf)
            test_val_acc = accuracy_score(val_set.Y,test_val_inf_out['labels'])
            logger.debug('Validation accuracy from epoch loop : {}'.format(best_model.val_acc))
            logger.debug('Validation accuracy after loaded : {}'.format(test_val_acc))

        return epoch_loss

    def get_validation_error(self,model,val_set,inf_conf):
        
        inf_out = ClassfierInference(logger=self.logger).predict(model,val_set,inf_conf)
        val_error = 1 - accuracy_score(val_set.Y,inf_out['labels'])
        return val_error 
