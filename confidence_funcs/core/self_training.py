
import sys 
sys.path.append('../')
sys.path.append('../../')
import copy 
from .query_strategies import * 
from .conf_defaults import *

#from .auto_labeling import *
from .pseudo_labeling import *

from .model_utils import * 

from src.data_layer.data_manager import *
from src.calibration.calibrators import * 
from .query_strategies.query_strategies_factory import * 


class SelfTraining:
    
    def __init__(self,conf, dm, logger=None):
        
        set_defaults(conf)

        self.conf   = conf 
        self.logger = logger
        self.dm     = dm 
        
        self.ds_unlbld     = dm.ds_std_train
        self.ds_std_val    = dm.ds_std_val
        self.ds_std_test   = dm.ds_std_test 
        
        self.N_v_std       = len(self.ds_std_val)

        self.auto_lbl_conf  = conf['auto_lbl_conf']

        self.ensure_val_frac = 0.2 

        self.random_seed        = conf['random_seed'] 
            
        self.per_epoch_out = []
        
        self.total_pts = len(self.ds_unlbld)
        self.cur_query_count = 0

        self.model_conf = conf['model_conf']

        self.epoch   = 0

        self.max_t = float('inf')
        self.num_classes = conf['data_conf']['num_classes']
        self.lst_classes = np.arange(0,self.num_classes)

        self.margin_thresholds = [self.max_t]*self.num_classes
        
        self.max_train_query = conf.train_pts_query_conf.max_num_train_pts

    
    def init(self):
        self.query_strategy = get_query_strategy(self.dm,None,self.conf,self.logger)
        # do any clustering etc. here.
    

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

        logger.debug('Validation Data Size :{}'.format(n_v))

    def query_training_batch(self,epoch_out):
        # query points to add in the training set.
        logger  = self.logger
        logger.debug('Querying next training batch')
        q_conf  = self.conf.train_pts_query_conf

        cur_avlbl_q_bgt = self.max_train_query - self.cur_query_count

        logger.debug(f'Current Available Query Budget: {cur_avlbl_q_bgt}')

        n_u             = self.dm.get_current_unlabeled_count(include_pseudo_labeled=False)
        
        #print(n_u,q_conf.query_batch_size)


        bs              = min(n_u, q_conf.query_batch_size, cur_avlbl_q_bgt)

        logger.debug(f'Query Batch Size = {bs}')

        self.query_strategy = get_query_strategy(self.dm,self.cur_clf,self.conf,self.logger)

        q_idx               = self.query_strategy.query_points(bs)
        
        self.dm.mark_queried(q_idx,round_id=self.epoch)

        logger.info(f'Queried {len(q_idx)} pts to add in training pool')

        epoch_out['query_points'] = q_idx
        self.cur_query_count += len(q_idx)


    def run_one_epoch(self):
        
        logger = self.logger 
        epoch  = self.epoch 
        conf   = self.conf
        q_conf  = self.conf.train_pts_query_conf


        epoch_out = {}
        logger.info('===========================================================================')
        logger.debug(f'========================= BEGIN EPOCH {epoch} ============================')
       # <<<<<<<<<<<<<<<<<<<<<<<<< BEGIN QUERYING POINTS BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<
        n_u = self.dm.get_current_unlabeled_count(include_pseudo_labeled=True)
        logger.debug('Number of unalabeled points  :{}'.format(n_u))

        if(epoch==0):
            logger.info('Querying first batches of training and validation samples.')
            self.query_seed_points(epoch_out)
            
        ## Enable if doing active self-training
        elif (q_conf['enable_active_querying']):
            self.query_training_batch(epoch_out)

        n_u = self.dm.get_current_unlabeled_count(include_pseudo_labeled=True)
        n_v = self.dm.get_current_validation_count()
        logger.debug('Validation Count For Current round {}'.format(n_v))
        
        epoch_out['num_val_pts'] = n_v
        

        logger.debug('Num Unlabeled Points After Querying :{}'.format(n_u))        
        
        #  >>>>>>>>>>>>>>>>>>>>>>>>>>> END QUERYING POINTS BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        if(n_v==0):
            return epoch_out
        
        cur_val_ds,cur_val_idcs  = self.dm.get_current_validation_data()
        epoch_out['begin_val_idcs']   = cur_val_idcs

        train_conf     = conf['training_conf']
        model_conf     = conf['model_conf']
        #<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN TRAINING BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<
        logger.info('===========================================================================')
        logger.info('========================== Begin Model Training ===========================')
        
        #train_from_scratch = train_conf['train_from_scratch'] or epoch>0 

        # train new model
        cur_train_ds, cur_train_idcs = self.dm.get_current_training_data(include_pseudo_labeled=True)

        logger.info(f'Training data size : {len(cur_train_ds)}')

        # check if there is just one class in training data
        bad_ds_flag = len(set(cur_train_ds.Y))==1
        if(bad_ds_flag):
            logger.error('Bad training dataset with only one class!!!')
            logger.error('Going to the next epoch to get more training points.')
            # go to next epoch and query more points.
            return epoch_out 
        
        ckpt_loaded = False
        if(epoch == 0 and conf.eval =="hyp"):
            logger.info("xxxxxxxx Loading pretrained model xxxxxxxxxx")
            
            ckpt_loaded = self.load_model_in_first_round_if_exists()

            if(ckpt_loaded):
                logger.info("xxxxxxxx Loaded pretrained model xxxxxxxxxx")
        

        if( not ckpt_loaded):
            logger.info('--------------- Begin Model Training ------------')
            
            logger.info('Training conf :{}'.format(train_conf))
            logger.info('Model conf : {}'.format(model_conf))

            self.cur_clf = train_model(cur_train_ds,conf.model_conf, conf.training_conf, conf.inference_conf, 
                                   logger,cur_val_ds)
            
            logger.info('--------------- End Model Training ------------')
        
        test_err     = get_test_error(self.cur_clf,self.dm.ds_std_test,conf['inference_conf'])
        train_err    = get_test_error(self.cur_clf,cur_train_ds, conf.inference_conf)
        #val_err = self.get_test_error(self.cur_clf,self.ds_std_val,self.conf['inference_conf'])

        epoch_out['train_error'] = train_err
        epoch_out['test_err']    = test_err    
        #epoch_out['val_error']  = val_err 
        

        logger.info(f'Training error of trained model : {train_err*100 :.2f}')
        logger.info(f'Test error of the model         : {test_err*100  :.2f}')

        logger.info('========================= End Model Training   =========================')
        logger.info('===========================================================================')


        if(train_conf['save_ckpt'] and train_conf['ckpt_save_path'] is not None and not ckpt_loaded):
                
            logger.info(f"Saving model checkpoint to path {train_conf['ckpt_save_path']}")
            self.save_state(train_conf['ckpt_save_path'])
            logger.info('Saved model checkpoint.')
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> END TRAINING BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>


        #<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN CALIBRATION BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<

        if( conf['calib_conf'] and conf['calib_conf']['type']=='post_hoc'):
            calib_conf    = conf['calib_conf'] 
            logger.info('========================= Training Post-hoc Calibrator   =========================')
            self.cur_calibrator  = get_calibrator(self.cur_clf,calib_conf,logger)
            
            # randomly split the current available validation points into two parts.
            # one part will be used for training the calibrator and other part for finding 
            # the auto-labeling thresholds.
            self.dm.select_calib_val_points(calib_frac=calib_conf['calib_val_frac'])
            cur_val_ds_nc , cur_val_idcs_nc  = self.dm.get_cur_non_calib_val_ds()
            cur_val_ds_c , cur_val_idcs_c    = self.dm.get_cur_calib_val_ds()
            logger.info(f"Number of validation points for training calibrator : {len(cur_val_ds_c)}")
            self.cur_calibrator.fit(cur_val_ds_c, ds_val_nc=cur_val_ds_nc)
        else:
            logger.info('=========================    No Post-hoc Calibration     =========================')
            self.cur_calibrator = None 
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>> END CALIBRATION BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>


        #<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN PSEUDO-LABELING BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<
        logger.info('==========================================================================')
        logger.info('========================= Begin Pseudo labeling Procedure ==================')

        # unmark previously psuedo-labeled points.
        self.dm.unmark_pseudo_labeled()
        
        # pseudo label all points that can be psuedo-labeled with this classifier
        pseudo_labeler = PseudoLabeling(conf,self.dm,self.cur_clf,logger,self.cur_calibrator)
        e_out          = pseudo_labeler.run(epoch=self.epoch)

        logger.info('========================= End Pseudo labeling Procedure  ===================')

        epoch_out.update(e_out)

        '''
        # Not removing for self-training

        if('val_idcs_to_rm' in e_out):
            val_idcs_to_rm = e_out['val_idcs_to_rm']
            logger.debug('Num Validation pts to remove : {}'.format(len(val_idcs_to_rm)))            
            
            if(val_idcs_to_rm is not None):
                self.dm.remove_validation_points(val_idcs_to_rm,round_id=epoch)
        '''

        n_v_end = self.dm.get_current_validation_count()

        if(n_v_end>0):
            cur_val_ds,val_idcs = self.dm.get_current_validation_data()
            epoch_out['end_val_idcs'] = val_idcs
        else:
            epoch_out['end_val_idcs'] = []

        #>>>>>>>>>>>>>>>>>>>>>>>>>>> END AUTO-LABELING BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>   
        
        #<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN SAVING STATE <<<<<<<<<<<<<<<<<<<<<<<<<
        if(self.conf.store_model_weights_in_mem):
            epoch_out['clf_weights'] = self.cur_clf.get_weights()
            epoch_out['cur_clf'] = copy.deepcopy(self.cur_clf)
                
        #>>>>>>>>>>>>>>>>>>>>>>>>>>> END SAVING STATE >>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        logger.debug('=============================== END Epoch {} ======================='.format(epoch))
        cur_counts = self.dm.get_pseudo_labeling_counts()
        epoch_out['cur_counts'] = cur_counts
        logger.info(cur_counts)

        return epoch_out 

    def run_al_loop(self):
        
        # using a validation set and self.cur_err > error_threshold

        #train_conf = self.conf['training_conf']
        
        conf = self.conf 
        logger = self.logger 
        self.epoch = 0
        lst_epoch_out = []
        self.lst_epoch_out = lst_epoch_out
        
        prev_q = 0
        if("eval" in conf and conf.eval == "hyp"):
            logger.info(f'xxxxxxxxxxxxxxxxxxxxx  Running TBAL with evaluation (auto-labeling) on hyp data  xxxxxxxxxxxxxxxxxxxxx')
        else:
            logger.info(f'xxxxxxxxxxxxxxxxxxxxx  Running TBAL with evaluation (auto-labeling) on the full unlabeled data   xxxxxxxxxxxxxxxxxxxxx')

        while(not self.check_stopping_criterion()):
            
            #self.cur_query_count += prev_q

            epoch_out = self.run_one_epoch()
            self.epoch+=1
            #print(epoch_out)

            lst_epoch_out.append(epoch_out)

        return lst_epoch_out
    
    def check_stopping_criterion(self):
        conf = self.conf 
        n_u = self.dm.get_current_unlabeled_count()
        labeled_all = n_u == 0
        
        self.logger.debug('Unlabeled count in check_stop_criterion {}'.format(n_u))
        self.logger.debug(f'cur_query_count= {self.cur_query_count} and max_query_count={self.max_train_query}')
        #print(labeled_all)
        if(conf.eval=="hyp" and self.epoch==1):
            return True

        if(labeled_all):
            # This conditions overrides all criterias.
            return labeled_all 
        
        if(conf.stopping_criterion=='label_all'):
            n_u = self.dm.get_current_unlabeled_count()
            return n_u == 0

        if(conf.stopping_criterion=='max_num_train_pts' ):
            
            return self.cur_query_count >= self.max_train_query

        if(conf.stopping_criterion=='max_epochs' ):
            return self.epoch >= self.conf['max_epochs']
        
        # implement other stopping criterions.. 

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
            # if self.model_conf['should_compile']:
            #     if not isinstance(self.cur_clf.model, torch._dynamo.eval_frame.OptimizedModule):
            #         self.cur_clf.model = torch.compile(self.cur_clf.model)
    
      
    def load_model_in_first_round_if_exists(self):
        
        assert self.epoch == 0

        logger         = self.logger
        conf           = self.conf 
        train_conf     = conf['training_conf']
        model_conf     = conf['model_conf']
        inference_conf = conf['inference_conf']
    
        ckpt_path = train_conf['ckpt_load_path']
        
        if(os.path.exists(ckpt_path)):
            self.cur_clf = clf_factory.get_classifier(model_conf,self.logger)
            logger.info('Loading model from path: {}'.format(ckpt_path))
            self.load_state(ckpt_path)
            return True 
        else:
            logger.info('Checkpoint path does not exist, training from scratch:')
            train_conf['train_from_scratch'] = True
            return False 
           