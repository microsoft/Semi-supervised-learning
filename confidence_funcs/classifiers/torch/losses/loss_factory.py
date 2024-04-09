import sys 
from .common_losses import * 


def get_loss_function(loss_conf):

    
    loss_fun_name = loss_conf['loss_function']

    if(loss_fun_name == 'std_cross_entropy'):
        return StdCrossEntropyLoss(loss_conf)
    
    elif(loss_fun_name == 'crl'):
        from .crl_loss import CRLLoss
        return CRLLoss(loss_conf)
    
    elif(loss_fun_name == 'squentropy'):
        return StdSquentropyLoss(loss_conf)
    
    elif(loss_fun_name == 'focal'):
        from .focal_loss import FocalLoss
        return FocalLoss(loss_conf)
    
    elif(loss_fun_name == 'focal_adaptive'):
        from .focal_loss import FocalLossAdaptive
        return FocalLossAdaptive(loss_conf)
    
    elif(loss_fun_name == 'mmce'):
        from .mmce_loss import MMCE
        return MMCE(loss_conf)

    elif(loss_fun_name == 'avuc'):
        from .avuc_loss import AUAvULoss
        return AUAvULoss(loss_conf)

    else:
        raise ValueError("Unknown loss function: %s"%loss_fun_name)