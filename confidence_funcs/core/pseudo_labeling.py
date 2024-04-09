import sys 
sys.path.append('../')
sys.path.append('../../')


import numpy as np 
from sklearn.metrics import accuracy_score

from collections import defaultdict
#from .human_labeling_helper import *
from .query_strategies import * 
from .conf_defaults import *
from .threshold_estimation import *


from src.classifiers import clf_factory 
from src.data_layer.data_manager import DataManager
from src.calibration.calibration_utils import * 
from src.calibration.calibrators import * 


class PseudoLabeling:
    
    def __init__(self,conf,dm,clf=None,logger=None,calibrator=None):
        
        set_defaults(conf)
        
        self.conf = conf 
        self.cur_clf   =  clf 
        self.logger = logger 
        self.dm = dm # data manager

        self.meta_df = dm.meta_df 

        self.ds_unlbld = dm.ds_std_train
        self.ds_val    = dm.ds_std_val
        self.ds_test   = dm.ds_std_test 
        
        self.random_seed = conf['random_seed']

        self.pseudo_lbl_conf = conf['pseudo_lbl_conf']

        self.pseudo_label_err_threshold = self.pseudo_lbl_conf['pseudo_label_err_threshold']

        self.max_t = float('inf')
        self.num_classes = self.dm.num_classes
        self.lst_classes = np.arange(0,self.num_classes)
        self.calibrator = calibrator
        self.margin_thresholds = [self.max_t]*self.num_classes

        self.pseudo_lbl_conf.setdefault('threshold_estimation',"val_estimate")
        
    
    def run(self,epoch=0):
        
        logger = self.logger 
        conf = self.conf

        epoch_out = {}
        pseudo_lbl_conf = self.pseudo_lbl_conf

        # Before doing anything first check if there are points to pseudo-label.
        if(conf.eval == 'hyp'):
            logger.info(f'xxxxxxxxxxxxxxxxxxxxx Pseudo-labeling hyp data  xxxxxxxxxxxxxxxxxxxxx')
            logger.info(f'xxxxxxxxxxxxxxxxxxxxx  Points in hyp data are treated as unlabeled xxxxxxxxxxxxxxxxxxxxx')
            cur_unlbld_idcs  = self.dm.get_current_unlabeled_hyp_idcs() 
            
        else:
            logger.info(f'xxxxxxxxxxxxxxxxxxxxx  Pseudo-labeling actual remaining unlabeled data  xxxxxxxxxxxxxxxxxxxxx')
            cur_unlbld_idcs  = self.dm.get_current_unlabeled_idcs()
        
        cur_unlbld_idcs = np.array(cur_unlbld_idcs)
        n_u = len(cur_unlbld_idcs)
        if(n_u==0):
            logger.info('No unlabeled points left, exiting..')
            return {}
        
        method_name = pseudo_lbl_conf['method_name']

        logger.info(f'========================= Begin Pseudo-Labeling {method_name} ==========================')
        logger.debug('Pseudo Labeling Conf : {}'.format(pseudo_lbl_conf))
        logger.info('Number of unlabeled points : {}'.format(n_u))

        epoch_out['unlabeled_pts_idcs'] = cur_unlbld_idcs
        epoch_out['num_unlabeled'] = n_u 
        
        # load checkpoint from the path given in the pseudo-label config.
        # TODO: Want to have data manager state, model and calibrator in the checkpoint

        if(self.cur_clf is None):
            # load from check point
            ckpt_load_path =pseudo_lbl_conf['ckpt_load_path']

            logger.info('Loading model checkpoint from :{}'.format(ckpt_load_path))
            self.load_state(ckpt_load_path)


        #cur_val_ds,cur_val_idcs = self.dm.get_validation_data()
        cur_val_ds,cur_val_idcs = self.dm.get_current_validation_data()

     
        cur_unlbld_ds = self.dm.get_subset_dataset(cur_unlbld_idcs)

        #unlbld_subset_ds = self.ds_unlbld.get_subset(unlbld_idcs)
        
        if(method_name=='all'):
            lst_pseudo_lbld_pts = self.pseudo_label_all(cur_unlbld_idcs,cur_unlbld_ds,epoch_out)
            
            epoch_out['val_idcs_to_rm'] = cur_val_idcs

        elif(method_name=='selective'):
            lst_pseudo_lbld_pts = self.selective_pseudo_label(cur_unlbld_idcs,cur_unlbld_ds,epoch_out)

        n_a = len(lst_pseudo_lbld_pts)
            
        epoch_out['pseudo_lbld_idx_lbl'] = lst_pseudo_lbld_pts
        epoch_out['num_pseudo_labeled'] = n_a

        # mark pseudo-labeled points
        self.dm.mark_pseudo_labeled(lst_pseudo_lbld_pts,round_id=epoch)

        logger.info('Num pseudo labeled points : {} '.format(n_a))

        val_idcs_to_rm = epoch_out['val_idcs_to_rm']
        #self.dm.remove_validation_points(val_idcs_to_rm,round_id=epoch)
        logger.info('Num validation pts to remove : {}'.format(len(val_idcs_to_rm)))
        
        logger.info('============================== Done Pseudo-Labeling ==============================')
        return epoch_out 
    
    def pseudo_label_all(self,cur_unlbld_idcs,cur_unlbld_ds,epoch_out):
        unlbld_inf_out = self.run_inference(self.cur_clf,cur_unlbld_ds,self.conf['inference_conf'],self.calibrator)
        epoch_out['lst_t_i'] = [0]*self.num_classes

        scores = unlbld_inf_out[self.pseudo_lbl_conf['score_type']]
        y_hat  = unlbld_inf_out['labels']

        n_u = len(cur_unlbld_idcs)
        selected_idcs = cur_unlbld_idcs

        lst_pseudo_lbld_pts = [{'idx':cur_unlbld_idcs[i],'label':int(y_hat[i]),'confidence':float(scores[i])} for i in range(n_u)  ]
        epoch_out['unlbld_inf_out'] = unlbld_inf_out

        return lst_pseudo_lbld_pts

    def selective_pseudo_label(self,cur_unlbld_idcs,cur_unlbld_ds,epoch_out):

        cur_val_ds_all, cur_val_idcs_all = self.dm.get_current_validation_data()
        cur_val_ds_nc , cur_val_idcs_nc  = self.dm.get_cur_non_calib_val_ds()
        cur_val_ds_c , cur_val_idcs_c    = self.dm.get_cur_calib_val_ds()
        
        #print(cur_val_idcs_c[:10], cur_val_idcs_nc[:10])
        
        n_v = len(cur_val_idcs_nc) 
        epoch_out['cur_num_val']    = len(cur_val_idcs_all) 
        epoch_out['cur_num_val_nc'] = len(cur_val_idcs_nc) 
        epoch_out['cur_num_val_c']  = len(cur_val_idcs_c) 

    
        logger = self.logger 
        logger.info('Using number of validation points : {}'.format(n_v))
        

        lst_t_val = []
        
        val_inf_out_nc = self.run_inference(self.cur_clf,cur_val_ds_nc,self.conf['inference_conf'],self.calibrator)
        
        val_inf_out_c  = None 

        if(cur_val_ds_c and len(cur_val_ds_c)>0):
            val_inf_out_c = self.run_inference(self.cur_clf,cur_val_ds_c,self.conf['inference_conf'],self.calibrator)
            epoch_out['val_inf_out_c'] = val_inf_out_c 
            val_inf_out_c['true_labels']  = cur_val_ds_c.Y 

        #val_inf_out_all = self.run_inference(self.cur_clf,cur_val_ds_all,self.conf['inference_conf'],self.calibrator)
        
        cal_out = compute_calibration(cur_val_ds_nc.Y, val_inf_out_nc['labels'], val_inf_out_nc['confidence'], num_bins=10)
        logger.debug(f"Expected Calibration Error on Validation set : {cal_out['expected_calibration_error']}")
        
        epoch_out['ECE_on_val'] = cal_out['expected_calibration_error']
        epoch_out['ECE_on_val_num_pts'] = epoch_out['cur_num_val_nc']

        epoch_out['calibration_output_on_val'] = cal_out 
        log_no_cal_ece = True 

        if(log_no_cal_ece and self.calibrator is not None):
            val_inf_out_nc_2 = self.run_inference(self.cur_clf,cur_val_ds_nc,self.conf['inference_conf'],None)
            no_cal_out = compute_calibration(cur_val_ds_nc.Y, val_inf_out_nc_2['labels'], val_inf_out_nc_2['confidence'], num_bins=10)
            logger.debug(f"Expected Calibration Error on Validation set with NO Calibration : {no_cal_out['expected_calibration_error']}")
            epoch_out['ECE_on_val_no_calib'] = no_cal_out['expected_calibration_error']
            epoch_out['ECE_on_val_no_calib_num_pts'] = len(cur_val_ds_nc)
            epoch_out['val_calibration_output_no_calib'] = no_cal_out 
        
        val_inf_out_nc['true_labels']  = cur_val_ds_nc.Y 
        epoch_out['val_inf_out_nc']    = val_inf_out_nc 

        err_threshold = self.pseudo_label_err_threshold

        th_estimation = self.pseudo_lbl_conf['threshold_estimation']
        val_idcs_to_rm = []
        if(th_estimation == "val_estimate"):
            logger.info('Determining Thresholds : Class Wise : {}'.format(self.pseudo_lbl_conf['class_wise']))
            logger.info('Using Pseudo-Labeling Error Threshold = {}'.format(err_threshold))

            lst_t_val, val_idcs_to_rm, val_err,cov = determine_threshold(self.lst_classes,val_inf_out_nc,
                                                                    self.pseudo_lbl_conf,cur_val_ds_nc,
                                                                    cur_val_idcs_nc,logger,err_threshold)
            
            logger.info('pseudo-labeling thresholds from val set: {}'.format(lst_t_val))
        
        elif(th_estimation=="fixed"):
            lst_t_val = [self.pseudo_lbl_conf["fixed_threshold"]]*self.num_classes

            lst2 = [cur_val_idcs_nc[i] for i in range(len(val_inf_out_nc['confidence'])) if val_inf_out_nc['confidence'][i] >=  
                    lst_t_val[ val_inf_out_nc['labels'][i] ] ]
            
            val_idcs_to_rm.extend(lst2)

            Y_val   = val_inf_out_nc['true_labels']
            y_hat   = val_inf_out_nc['labels']
            val_err = 1-accuracy_score(Y_val,y_hat)

            logger.info('using fixed pseudo-labeling thresholds : {}'.format(lst_t_val))
        else:
            logger.error('undefinded threshold estimation procedure.')
        
        if(val_inf_out_c):
            lst2 = [cur_val_idcs_c[i] for i in range(len(val_inf_out_c['confidence'])) if val_inf_out_c['confidence'][i] >=  
                    lst_t_val[ val_inf_out_c['labels'][i] ] ]
            #epoch_out['val_idcs_to_rm'].extend(lst2)
            val_idcs_to_rm.extend(lst2)
        
        epoch_out['val_idcs_to_rm'] = val_idcs_to_rm
        epoch_out['val_err'] = val_err 

        unlbld_inf_out = self.run_inference(self.cur_clf,cur_unlbld_ds,self.conf['inference_conf'],self.calibrator)
        
        record_ece_on_unlbld_pool = False 

        if(record_ece_on_unlbld_pool):
            unlbld_cal_out = compute_calibration(cur_unlbld_ds.Y, unlbld_inf_out['labels'], unlbld_inf_out['confidence'], num_bins=10)
            logger.debug(f"Expected Calibration Error on Unlabeled set : {unlbld_cal_out['expected_calibration_error']}")
            epoch_out['ECE_on_unlbld_calib'] = unlbld_cal_out['expected_calibration_error']
            epoch_out['ECE_on_unlbld_calib_num_pts'] = len(unlbld_cal_out)
            epoch_out['unlbld_calibration_output_no_calib'] = unlbld_cal_out 

        epoch_out['unlbld_inf_out'] = unlbld_inf_out
        epoch_out['lst_t_i_val']    = lst_t_val
        
        scores = unlbld_inf_out[self.pseudo_lbl_conf['score_type']]
        y_hat = unlbld_inf_out['labels']
        
        #print(sum(scores>0.95))
        #print(max(scores))

        lst_t_val = np.array(lst_t_val) 
        lst_pseudo_lbld_pts = []
        n = len(cur_unlbld_idcs)
        
        # check if the score is bigger than the threshold for the predicted class.
        selected_idcs = [ i for i in range(n) if scores[i]>=lst_t_val[y_hat[i]] ]

        lst_pseudo_lbld_pts = [{'idx':cur_unlbld_idcs[i],'label':int(y_hat[i]),'confidence':float(scores[i])} for i in selected_idcs  ]
        return lst_pseudo_lbld_pts
    
    def save_state(self,path):
        torch.save({ 'model_state_dict': self.cur_clf.model.state_dict(),
                    'conf':self.conf,
                    'meta_df':self.meta_df 
                    }, path)
        
    def load_state(self,path):
        checkpoint = torch.load(path)
        self.cur_clf = clf_factory.get_classifier(self.conf['model_conf'],self.logger)

        self.cur_clf.model.load_state_dict(checkpoint['model_state_dict'])
    
    def run_inference(self,clf, ds, inference_conf, calibrator=None):

        if(calibrator is not None):
            inf_out = calibrator.predict(ds,inference_conf)
            logger.debug('Got Calibrated Outputs...')                
        else:
            inf_out = clf.predict(ds,inference_conf)
        
        return inf_out