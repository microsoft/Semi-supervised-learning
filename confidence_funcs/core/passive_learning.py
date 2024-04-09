import sys

sys.path.append('../')

import torch
from sklearn.manifold import TSNE
from .query_strategies import * 
from src.classifiers import clf_factory

from .conf_defaults import *
import copy 
import numpy as np 
from .model_utils import * 
import os 

class PassiveLearning:
    
    def __init__(self,conf, dm,logger=None):
                
        set_defaults(conf)

        self.conf   = conf 
        self.logger = logger
        self.dm     = dm 
        
        self.ds_unlbld     = dm.ds_std_train
        self.ds_std_val    = dm.ds_std_val
        self.ds_std_test   = dm.ds_std_test 
        
        self.N_v_std       = len(self.ds_std_val)

        self.num_classes =self.dm.ds_std_train.num_classes
        self.lst_classes = np.arange(0,self.num_classes,1)
        self.logger = logger
        self.cur_query_count = 0
        self.ckpt_loaded = False 
        
        self.max_train_query = conf.train_pts_query_conf.max_num_train_pts


    def query_seed_points(self,epoch_out):

        logger = self.logger 
        seed_train_size  = self.conf.train_pts_query_conf.seed_train_size

        logger.debug('Querying {} seed training points'.format(seed_train_size))
        q_idx = self.dm.select_seed_train_points(k=seed_train_size,method='randomly')
        
        epoch_out['seed_train_pts'] = q_idx 
        epoch_out['query_points']   = q_idx
        self.cur_query_count       += len(q_idx)

        logger.debug('Queried {} seed points for training'.format(len(q_idx)))

        #n_v                        = int(self.N_v_std * self.val_frac_for_auto_lbl)
        val_query_conf = self.conf.val_pts_query_conf 
        n_v            = val_query_conf.max_num_val_pts

        cur_val_idcs               = self.dm.query_validation_points(n_v,method='random')
        cur_val_ds,cur_val_idcs    = self.dm.get_current_validation_data()
        epoch_out['seed_val_pts']  = cur_val_idcs

        logger.debug('Validation Data Size :{}'.format(len(cur_val_idcs)))

        remaining_train_pts = self.max_train_query - seed_train_size
        logger.debug(f'Querying rest of the training points {remaining_train_pts} in single batch')
        
        q_idx = self.dm.select_seed_train_points(k=remaining_train_pts,method='randomly')


    def run(self):

        
        logger         = self.logger
        conf           = self.conf 
        train_conf     = conf['training_conf']
        model_conf     = conf['model_conf']
        inference_conf = conf['inference_conf']
        self.inference_conf = inference_conf

        out = {}
        epoch = 0

        self.query_seed_points(out)

        cur_val_ds,cur_val_idcs  = self.dm.get_current_validation_data()
        cur_train_ds, cur_train_idcs = self.dm.get_current_training_data()

        
        n_t = len(cur_train_ds)

        logger.info('Labeled data size for training: {}'.format(n_t))
        logger.info('Labeled data size for validation: {}'.format(len(cur_val_idcs)))

        if(train_conf['ckpt_load_path'] == None or train_conf['train_from_scratch'] ==True  ):
            # create a new model for training. make it part of config...            
            self.cur_clf = clf_factory.get_classifier(model_conf,self.logger)
            train_conf['train_from_scratch'] = True
        else:
            ckpt_path = train_conf['ckpt_load_path']
            
            if(os.path.exists(ckpt_path)):
                logger.info('Loading model from path: {}'.format(ckpt_path))
                self.load_state(ckpt_path)
                self.ckpt_loaded = True 
            else:
               logger.info('Checkpoint path does not exist, training from scratch:')
               train_conf['train_from_scratch'] = True

        
        if( train_conf['train_from_scratch']):
            logger.info('--------------- Begin Model Training ------------')
            
            logger.info('Training conf :{}'.format(train_conf))
            logger.info('Model conf : {}'.format(model_conf))

            self.cur_clf = train_model(cur_train_ds,model_conf, train_conf,
                                       conf.inference_conf,logger,cur_val_ds = cur_val_ds)
            
            logger.info('--------------- End Model Training ------------')

        test_err = get_test_error(self.cur_clf,self.dm.ds_std_test,conf['inference_conf'])

        logger.info(f'Test error of the model : {test_err*100 :.2f}')

        if(train_conf['save_ckpt'] and train_conf['ckpt_save_path'] is not None):

            if( not self.ckpt_loaded):  # temporary to avoid re-saving a loaded ckpt.
                
                logger.info('Saving model checkpoint to path')
                self.save_state(train_conf['ckpt_save_path'])
                logger.info('Saved model checkpoint to path')
        
        if(conf['store_model_weights_in_mem']):
            out['clf_weights'] = self.cur_clf.get_weights()

        return out 
    
    def save_state(self,path):
        if(self.conf['model_conf']['lib']=='pytorch'):
            model_state_dict = self.cur_clf.model.state_dict()
        else:
            model_state_dict =None 
            
        torch.save({ 'model_state_dict': model_state_dict,
                        'conf':self.conf,
                        'meta_df':self.dm.meta_df 
                    }, path)
        
    def load_state(self,path):
        
        checkpoint = torch.load(path)
        if(self.conf['model_conf']['lib']=='pytorch'):
            self.cur_clf = clf_factory.get_classifier(self.conf['model_conf'],self.logger)
            self.cur_clf.model.load_state_dict(checkpoint['model_state_dict'])
        
        #self.dm.meta_df = checkpoint['meta_df']  