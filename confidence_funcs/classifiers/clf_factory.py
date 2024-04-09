import sys
sys.path.append('../')
sys.path.append('../../')
#sys.path.append('../../../')

from .torch.pytorch_clf import *
from .sklearn.sklearn_clf import *
from .logistic_regression import *

def get_classifier(model_conf,logger=None):
    # conf should have, lib = sklearn or torch
    # and model_conf, train_conf, inference_conf
    logger = logger 
    if(model_conf['lib']=='pytorch'):
        clf = PyTorchClassifier(model_conf,logger=logger)
    elif(model_conf['lib']=='sklearn'):
        clf = SkLearnClassifier(model_conf,logger=logger)
    elif(model_conf['lib']=='custom'):
        if(model_conf['model_name']=='logistic_regression'):
            clf = CustomLogisticRegression(model_conf,logger=logger)
    else:
        logger.info('invalid lib')
    
    return clf
    
    