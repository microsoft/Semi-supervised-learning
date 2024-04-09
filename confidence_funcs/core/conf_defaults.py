
def set_defaults(conf):

    '''
    TOODO: describe parameters and options
    Add new params with default values so that existing stuff works.
    '''
    conf.setdefault('random_seed',42)
    conf.setdefault("device","cpu")
    
    conf.setdefault("inference_conf",{"device":"cpu","shuffle":False})

    conf.setdefault('active_learning_conf',{})

    conf.setdefault("store_model_weights_in_mem",False)
    conf.setdefault("dump_result",False)

    train_conf = conf['training_conf']
    train_conf.setdefault('device',conf['device'])
    train_conf.setdefault('store_embedding',False)
    train_conf.setdefault('ckpt_load_path',None)
    train_conf.setdefault('save_ckpt',False)

    train_conf.setdefault('num_trials',1) 
    # need to use this for xor, since svm doesn't 
    # give good solution sometimes.
    
    if(conf['inference_conf'] is None):
        conf['inference_conf'] = {}

    conf['inference_conf'].setdefault('device',conf['device'])

    conf['active_learning_conf'].setdefault('score_type','confidence')

    conf.setdefault('calibrate_clf',False)
    conf.setdefault('calibration_conf',{})

    conf.setdefault('auto_lbl_conf',{})
    auto_lbl_conf = conf['auto_lbl_conf']
    auto_lbl_conf.setdefault('C_1',0.01)
    auto_lbl_conf.setdefault('fast',True) 
    auto_lbl_conf.setdefault('auto_label_err_threshold',0.05) # 5%
    auto_lbl_conf.setdefault('score_type','confidence')
    auto_lbl_conf.setdefault('class_wise','independent') 
    auto_lbl_conf.setdefault('ucb','hoeffding') 



    conf.setdefault('val_err_threshold',0.0)

