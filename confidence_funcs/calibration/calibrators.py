import sys 
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict 

from .HB_utils import *

from .auto_labeling_optimization_v0 import *
#from .auto_labeling_optimization_v1 import *
#from .auto_labeling_optimization_v2 import * 

#from .auto_labeling_optimization_v0_cooper import * 
 
from .abstract_calibrator import * 

from confidence_funcs.classifiers.torch.pytorch_clf import *
from confidence_funcs.data_layer.datasets.torch_dataset import * 

from confidence_funcs.data_layer.dataset_utils import * 

def get_calibrator(clf,calib_conf,logger):
    
    calib_name = calib_conf['name']
    logger.info(f'Creating instance of calibrator : {calib_name}')
    
    if(calib_name=='temp_scaling'):
        return TemperatureScalingCalibrator(clf,calib_conf,logger)
    
    if(calib_name=='histogram_binning'):
        logger.info('start histogram binning')
        return HistogramBinningCalibrator(clf,calib_conf,logger)
    
    if(calib_name=='histogram_binning_top_label'):
        logger.info('start histogram binning top label')
        return HB_toplabel(clf,calib_conf,logger)
    
    if(calib_name == 'auto_label_opt_v0'):
        logger.info('using auto-label-opt-v0 to learn g')
        return AutoLabelingOptimization_V0(clf, calib_conf,logger)
    
    if calib_name == 'scaling_binning':
        logger.info('start scaling binning')
        return ScalingBinningCalibrator(clf, calib_conf, logger)

    if calib_name == 'beta':
        logger.info('start beta calibration')
        return BetaCalibrator(clf, calib_conf, logger)
    if calib_name == 'dirichlet':
        logger.info('start dirichlet calibration')
        return DirichletCalibrator(clf, calib_conf, logger)
    else:
        logger.error(f'Unsupported Calibrator : {calib_name}')
        return None


class IdentityCalibrator(AbstractCalibrator):

    def __init__(self,clf,calib_conf,logger):
        self.clf = clf 
        self.calib_conf = calib_conf 
        self.logger    = logger 

    def fit(self, calib_ds,calib_ds_inf_out=None):
        pass 

    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        inf_out = clf_inference.predict(self.clf.model,dataset,inference_conf)
        return inf_out 

class TemperatureScalingCalibrator(AbstractCalibrator):

    def __init__(self,clf,calib_conf,logger):
        '''
        Takes calibration config as input 
        '''
        self.clf = clf
        self.calib_conf = calib_conf 
        self.calib_clf = PyTorchClassifier(model_conf={"name":"temp_scaling"},logger=logger)
        self.logger = logger 
        # if val_inf_out is none, then run inference here.

    def fit(self,calib_ds,calib_ds_inf_out=None,ds_val_nc=None):

        if(calib_ds_inf_out is None):
            clf_inference = ClassfierInference(self.logger)
            calib_ds_inf_out = clf_inference.predict(self.clf.model,calib_ds,None)
            
        logits = calib_ds_inf_out['logits'] 
        # logits should be nxk matrix. ( n : num data points, k: classes)
        #Y_ = (calib_ds.Y == calib_ds_inf_out['labels']).long() 
        Y_ = calib_ds.Y
        logits_ds = CustomTensorDataset(X=logits,Y=Y_)
        
        self.calib_clf.fit(logits_ds,self.calib_conf['training_conf'])
    
    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        inf_out = clf_inference.predict(self.clf.model,dataset,inference_conf)
        n= len(inf_out['logits'])
        logits_ds = CustomTensorDataset(X=inf_out['logits'],Y=torch.zeros(n))
        return clf_inference.predict(self.calib_clf.model,logits_ds,inference_conf)

class HistogramBinningCalibrator(AbstractCalibrator):

    def __init__(self,clf,calib_conf,logger):
        '''
        Takes calibration config as input 
        '''
        self.clf = clf
        self.calib_conf = calib_conf 
        #self.calib_clf = PyTorchClassifier(model_conf={"model_name":"temp_scaling"},logger=logger)
        self.logger = logger 
        # if val_inf_out is none, then run inference here.

    def fit(self,calib_ds,calib_ds_inf_out=None, ds_val_nc=None):

        if(calib_ds_inf_out is None):
            clf_inference = ClassfierInference(self.logger)
            calib_ds_inf_out = clf_inference.predict(self.clf.model,calib_ds,None)
            
        #logits = calib_ds_inf_out['logits'] 
        # logits should be nxk matrix. ( n : num data points, k: classes)
        #logits_ds = CustomTensorDataset(X=logits,Y=calib_ds.Y)
        num_bins = self.calib_conf['num_bins']
        binning_type  = self.calib_conf['binning_type']
        
        n_classes = self.clf.num_classes
        delta = 1/num_bins 
        self.delta = delta 
        self.num_bins = num_bins

        if(binning_type=='uniform_binning'):
            confidences = calib_ds_inf_out['confidence']
            probs = calib_ds_inf_out['probs']
            Y = calib_ds.Y 
            D = defaultdict()

            for i,p in enumerate(probs):
                for c in range(n_classes):
                    if(c in D):
                        D_c = D[c]
                    else:
                        D[c] = defaultdict(list)
                        D_c = D[c]
                    
                    #print(p[c],(p[c]/delta).long().item())

                    bin = min(num_bins-1, (p[c]/delta).long().item())
                    D_c[bin].append(1 if c==Y[i] else 0)

            for c in range(n_classes):
                D_c = D[c]
                for bin in range(num_bins):
                    #print(c,bin,D_c[bin])

                    if(len(D_c[bin])>0):
                        D_c[bin] = sum(D_c[bin])/len(D_c[bin])
                    else:
                        D_c[bin]= 0
        self.D = D 
        #self.calib_clf.fit(logits_ds,self.calib_conf['training_conf'])

    
    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        inf_out = clf_inference.predict(self.clf.model,dataset,inference_conf)
        n= len(inf_out['logits'])
        #logits_ds = CustomTensorDataset(X=inf_out['logits'],Y=torch.zeros(n))
        #clf_inference.predict(self.calib_clf.model,logits_ds,inference_conf)
        
        for i, p_hat in enumerate(inf_out['probs']):
            for c in range(self.clf.num_classes):
                bin = min(self.num_bins-1, (p_hat[c]/self.delta).long().item() )
                #print(self.D[c][bin])
                inf_out['probs'][i][c] = self.D[c][bin]

            inf_out['confidence'][i] = max(inf_out['probs'][i])
        
        return inf_out 


# https://github.com/AIgen/df-posthoc-calibration/blob/main/calibration.py
class HB_toplabel(AbstractCalibrator):
    #def __init__(self, calib_conf,logger):
    def __init__(self,clf, calib_conf, logger):
        ### Hyperparameters
        self.clf = clf
        self.calib_conf = calib_conf
        self.logger = logger
        self.points_per_bin = calib_conf["points_per_bin"]

        ### Parameters to be learnt 
        self.hb_binary_list = []
        
        ### Internal variables
        self.num_classes = None
    
    def fit(self, calib_ds,calib_ds_inf_out=None, ds_val_nc=None):
        
        #clf_inference = ClassfierInference(self.logger)
        #calib_ds_inf_out = clf_inference.predict(self.clf.model,calib_ds,None)
        calib_ds_inf_out = self.clf.predict(calib_ds)

        pred_mat = np.array(calib_ds_inf_out['probs'])
        y = np.array(list(calib_ds.Y))

        self.logger.info("Fitting histogram binning calibration, pred_mat.shape = " + str(pred_mat.shape) + ", y.shape = " + str(np.shape(y)))
    
        assert(self.points_per_bin is not None), "Points per bins has to be specified"
        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        y = y.squeeze()
        assert(pred_mat.shape[0] == y.size), "Check dimensions of input matrices"
        self.num_classes = pred_mat.shape[1]
        assert(np.min(y) >= 0 and np.max(y) < self.num_classes), "Labels should be numbered 0 ... L-1, where L is the number of columns in the prediction matrix"
        
        top_score = np.max(pred_mat, axis=1).squeeze()
        pred_class = (np.argmax(pred_mat, axis=1)).squeeze()
        
        for l in range(0, self.num_classes, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)

            bins_l = np.floor(n_l/self.points_per_bin).astype('int')

            self.logger.info("fit bin boundary/bias, class:{}, # of points:{},# of bins:{}".format(l, n_l, bins_l))

            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               self.logger.info("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}.".format(l, self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l,logger=self.logger)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict(self, dataset,inference_conf):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        
        #clf_inference = ClassfierInference(self.logger)
        #inf_out = clf_inference.predict(self.clf.model,dataset,inference_conf)
        inf_out = self.clf.predict(dataset)

        pred_mat = np.array(inf_out['probs'])

        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        assert(self.num_classes == pred_mat.shape[1]), "Number of columns of prediction matrix do not match number of labels"

        top_score = np.max(pred_mat, axis=1).squeeze()
        pred_class = (np.argmax(pred_mat, axis=1)).squeeze()

        n = pred_class.size
        
        if(n==1):
            pred_class = np.array([pred_class])
            top_score = np.array([top_score])

        pred_top_score = np.zeros((n))

        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]].predict(top_score[i])
        inf_out['bin_bdy_bias'] =  [[hb.bin_upper_edges,hb.mean_pred_values] for hb in self.hb_binary_list] # store the bin boundaries and mean predictions for each class
        inf_out['confidence'] = pred_top_score # store the recalibrated confidence scores
        return inf_out 
        
        

    def fit_top(self, top_score, pred_class, y):
        assert(self.points_per_bin is not None), "Points per bins has to be specified"

        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        y = y.squeeze()

        assert(min(np.min(y), np.min(pred_class)) >= 1), "Labels should be numbered 1 ... L, use HB_binary for a binary problem"
        assert(top_score.size == y.size), "Check dimensions of input matrices"
        assert(pred_class.size == y.size), "Check dimensions of input matrices"
        assert(y.size >= self.n_bins), "Number of bins should be less than the number of calibration points"

        self.num_classes = max(np.max(y), np.max(pred_class))
        
        for l in range(0, self.num_classes, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)
            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            
            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               self.logger.info("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}".format(self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict_top(self, top_score, pred_class):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        assert(top_score.size == pred_class.size), "Check dimensions of input matrices"
        assert(np.min(pred_class) >= 1 and np.min(pred_class) <= self.num_classes), "Some of the predicted labels are not in the range of labels seen while calibrating"
        n = pred_class.size
        pred_top_score = np.zeros((n))
        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]].predict(top_score[i])

        return pred_top_score


class HB_binary(AbstractCalibrator):
    def __init__(self, n_bins, logger):
        ### Hyperparameters
        self.delta = 1e-10
        self.logger = logger
        self.n_bins = n_bins

        ### Parameters to be learnt 
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None

        ### Internal variables
        self.fitted = False
        
    def fit(self, y_score, y):
        
        assert(self.n_bins is not None), "Number of bins has to be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
       
        assert(y_score.size == y.size), "Check dimensions of input matrices"
        assert(y.size >= self.n_bins), "Number of bins should be less than the number of calibration points"
        
        ### All required (hyper-)parameters have been passed correctly
        ### Uniform-mass binning/histogram binning code starts below

        # delta-randomization
        y_score = nudge(y_score, self.delta)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges = get_uniform_mass_bins(y_score, self.n_bins)

        # assign calibration data to bins
        bin_assignment = bin_points(y_score, self.bin_upper_edges)

        # compute bias of each bin 
        self.num_calibration_examples_in_bin = np.zeros([self.n_bins, 1])
        self.mean_pred_values = np.empty(self.n_bins)
        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            self.num_calibration_examples_in_bin[i] = sum(bin_idx)

            # nudge performs delta-randomization
            if (sum(bin_idx) > 0):
                self.mean_pred_values[i] = nudge(y[bin_idx].mean(),
                                                 self.delta)
            else:
                self.mean_pred_values[i] = nudge(0.5, self.delta)

        # check that my code is correct
        assert(np.sum(self.num_calibration_examples_in_bin) == y.size)

        # histogram binning done
        self.fitted = True

    def predict(self, y_score):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        y_score = y_score.squeeze()

        # delta-randomization
        y_score = nudge(y_score, self.delta)
        
        # assign test data to bins
        y_bins = bin_points(y_score, self.bin_upper_edges)
            
        # get calibrated predicted probabilities
        y_pred_prob = self.mean_pred_values[y_bins]
        return y_pred_prob


class ScalingBinningCalibrator(AbstractCalibrator):

    def __init__(self, clf, calib_conf, logger):
        self.clf = clf
        self.calib_conf = calib_conf 
        self.calib_clf = PyTorchClassifier(
            model_conf={"name": "temp_scaling"},
            logger=logger)
        self.logger = logger
        self.n_bins = calib_conf['training_conf']['num_bins']

    def fit(self, calib_ds, calib_ds_inf_out=None, ds_val_nc=None):
        # split the recalibration set into 3 parts, t1, t2, and t3
        indices = np.arange(len(calib_ds))
        size = len(calib_ds) // 3
        t1 = calib_ds.get_subset(indices[:size])
        t2 = calib_ds.get_subset(indices[size:2 * size])
        t3 = calib_ds.get_subset(indices[2 * size:])

        # step 1, function fitting on t1
        if calib_ds_inf_out is None:
            clf_inference = ClassfierInference(self.logger)
            calib_ds_inf_out = clf_inference.predict(self.clf.model,
                                                     t1,
                                                     None)
        logits_ds = CustomTensorDataset(X=calib_ds_inf_out['logits'],Y=t1.Y)
        self.calib_clf.fit(logits_ds,self.calib_conf['training_conf'])

        # step 2, binning scheme construction
        logits = clf_inference.predict(self.clf.model, t2, None)['logits']
        logits_ds = CustomTensorDataset(X=logits, Y=torch.zeros(len(logits)))
        scores = clf_inference.predict(
            self.calib_clf.model, logits_ds, None)['confidence']
        scores_sorted = np.sort(scores)
        groups = np.array_split(scores_sorted, self.n_bins)

        #print(set(scores_sorted))
        #print(list(map(lambda x: sorted(x, reverse=True)[:3], groups)))
        
        self.bin_upper_edges = np.array(list(map(lambda x: max(x, default=scores_sorted[-1]), groups)))

        # step 3, discretization
        logits = clf_inference.predict(self.clf.model, t3, None)['logits']
        logits_ds = CustomTensorDataset(X=logits, Y=torch.zeros(len(logits)))
        scores = clf_inference.predict(self.calib_clf.model, logits_ds, None)['confidence']
        sorted_scores = np.sort(scores)

        scale_bin = []

        running_set = []
        bin_id = 0
        
        for s in sorted_scores:
            if bin_id == len(self.bin_upper_edges):
                running_set.append(s) 
                continue
            if s <= self.bin_upper_edges[bin_id]:
                running_set.append(s) 
            else:
                scale_bin.append(np.mean(running_set))
                running_set = [s]
                bin_id += 1
        scale_bin.append(np.mean(running_set))

        for _ in range(len(scale_bin), len(self.bin_upper_edges)):
            scale_bin.append(scale_bin[-1])

        self.scale_bin = np.array(scale_bin)
    
    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        inf_out = clf_inference.predict(self.clf.model, dataset, inference_conf)
        n = len(inf_out['logits'])
        logits_ds = CustomTensorDataset(X=inf_out['logits'], Y=torch.zeros(n))
        scores = clf_inference.predict(self.calib_clf.model, logits_ds, inference_conf)['confidence']
        # find the index of the first element in bin_upper_edges that is larger than s
        bin_ids = np.searchsorted(self.bin_upper_edges, scores)
        bin_ids[bin_ids == len(self.scale_bin)] = len(self.scale_bin) - 1
        inf_out['confidence'] = self.scale_bin[bin_ids]
        return inf_out

class BetaCalibrator(AbstractCalibrator):
    def __init__(self, clf, calib_conf, logger):
        from betacal import BetaCalibration
        self.clf = clf
        self.calib_conf = calib_conf 
        self.calib_clf = BetaCalibration()
        self.logger = logger

    def fit(self, calib_ds, calib_ds_inf_out=None, ds_val_nc=None):
        if calib_ds_inf_out is None:
            clf_inference = ClassfierInference(self.logger)
            calib_ds_inf_out = clf_inference.predict(self.clf.model,
                                                     calib_ds,
                                                     None)
        self.calib_clf.fit(calib_ds_inf_out['confidence'].reshape(-1, 1), calib_ds.Y)

    
    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        inf_out = clf_inference.predict(self.clf.model, dataset, inference_conf)
        inf_out['confidence'] = self.calib_clf.predict(inf_out['confidence'].reshape(-1, 1))
        return inf_out


class DirichletCalibrator(AbstractCalibrator):
    def __init__(self, clf, calib_conf, logger):
        from .dirichletcal.calib import fulldirichlet
        self.clf = clf
        self.calib_conf = calib_conf 
        self.reg = calib_conf['training_conf']['reg']
        self.calib_clf = fulldirichlet.FullDirichletCalibrator(reg_lambda=self.reg, reg_mu=None, optimizer='fmin_l_bfgs_b')
        self.logger = logger

    def fit(self, calib_ds, calib_ds_inf_out=None, ds_val_nc=None):
        if calib_ds_inf_out is None:
            clf_inference = ClassfierInference(self.logger)
            calib_ds_inf_out = clf_inference.predict(self.clf.model,
                                                     calib_ds,
                                                     None)
        if isinstance(calib_ds.Y, list):
            self.calib_clf.fit(calib_ds_inf_out['probs'].numpy(), np.array(calib_ds.Y))
        else:
            self.calib_clf.fit(calib_ds_inf_out['probs'].numpy(), calib_ds.Y.numpy())

    
    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        inf_out = clf_inference.predict(self.clf.model, dataset, inference_conf)
        inf_out['confidence'] = self.calib_clf.predict(inf_out['probs'].numpy()).max(axis=1).reshape(-1)
        return inf_out
