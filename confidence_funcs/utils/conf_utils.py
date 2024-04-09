import copy
import sys 
import os 
from omegaconf import OmegaConf

def create_sub_confs(sub_conf,params,sub_conf_root):
    from itertools import product
    if(len(params)==0):
        sub_conf['fpath_key'] = sub_conf_root
        return [sub_conf] 

    keys, values = zip(*params.items())
    
    lst_confs = []
    for bundle in product(*values):
        d = dict(zip(keys, bundle))
        
        sub_conf_2 = copy.deepcopy(sub_conf)
        
        for k in d.keys():
            if("." in k):
                s = k.split(".")
                if(len(s)==2):
                    sub_conf_2[s[0]][s[1]] = d[k]
                if(len(s)==3):
                    sub_conf_2[s[0]][s[1]][s[2]] = d[k]
            else:
                sub_conf_2[k] = d[k]
        

        #conf['run_dir'] = '/'.join([ f'{k}__{d2[k]}' for k in sorted(d2.keys())])

        sub_conf_2['fpath_key'] =   os.path.join(* [ f'{k}__{d[k]}' for k in sorted(d.keys())  ]) # "/".join()
        sub_conf_2['fpath_key'] = sub_conf_root + '/' + sub_conf_2['fpath_key']
        lst_confs.append(sub_conf_2)
    return lst_confs

def get_model_ckpt_file_name(conf, keys= None):
    
    if(keys is None):
        keys = ['loss_function','optimizer','learning_rate','batch_size','momentum','weight_decay','max_epochs']
    
    s = "__".join([f"{k}_{conf.training_conf[k]}" for k in keys])
    
    N_t = conf.train_pts_query_conf.max_num_train_pts 

    model_ds = f"{conf.model_conf.name }_{conf.data_conf.name}"

    s = f"{model_ds}__Nt_{N_t}__{s}.ckpt"

    return s 

def create_confs(conf,params, should_filter_bigger_Nv=False):
    from itertools import product
    keys, values = zip(*params.items())
    lst_confs = []
    
    path_keys = ['method','C','C_1','eps','max_num_train_pts','max_num_val_pts','num_hyp_val_samples', 'seed_frac', 'query_batch_frac', 'training_conf', 'calib_conf','seed']
    std_train_conf_path_keys = ['optimizer','learning_rate', 'max_epochs', 'weight_decay']


    for bundle in product(*values):
        d = dict(zip(keys, bundle))

        if should_filter_bigger_Nv:
            if d['max_num_val_pts'] > d['max_num_train_pts']:
                continue
        
        conf_cp = copy.deepcopy(conf)
        n_q = d['max_num_train_pts']

        seed_size        = int(n_q*d['seed_frac'])
        query_batch_size = int(n_q*d['query_batch_frac'])

        if(conf_cp['data_conf']["name"] == "synth_concenteric_circles"):
            seed_size = 50
            query_batch_size = 25
            d['C'] = 10


        conf_cp['method']      = d['method'] 
        conf_cp["random_seed"] = d['seed']
        conf_cp['calib_conf']  = d['calib_conf']

        
        conf_cp['data_conf']['num_hyp_val_samples'] = d['num_hyp_val_samples']
        
        conf_cp['train_pts_query_conf']['max_num_train_pts'] = d['max_num_train_pts']
        conf_cp['train_pts_query_conf']['seed_train_size'] = seed_size
        conf_cp['train_pts_query_conf']['query_batch_size'] = query_batch_size
        conf_cp['val_pts_query_conf']['max_num_val_pts'] = d['max_num_val_pts']
        conf_cp['auto_lbl_conf']['C_1'] = d['C_1'] 
        conf_cp['auto_lbl_conf']['auto_label_err_threshold']= d['eps']

        if(conf_cp['method'] in ['tbal','active_learning']):
            conf_cp['train_pts_query_conf']['margin_random_v2_constant'] = d['C']
            #conf['train_pts_query_conf']['seed_train_size'] = seed_size
            #conf['train_pts_query_conf']['query_batch_size'] = query_batch_size
            conf_cp['stopping_criterion']= "max_num_train_pts"
        
        d2 = copy.deepcopy(d)

        if('training_conf' not in d):
            d['training_conf'] = conf_cp.training_conf
        
        else:
            conf_cp.training_conf = d['training_conf'] 

        if(d['calib_conf']is not None):
            calib_conf = conf_cp['calib_conf']
            calib_conf['auto_lbl_conf'] = conf_cp['auto_lbl_conf']
            
            # update the training_conf params by the the values in the calib conf's train conf.
            if(calib_conf['type']=='train_time'):
                pass 
                '''
                calib_train_conf = calib_conf['training_conf']
                for k in calib_train_conf.keys():
                    conf_cp['training_conf'][k] = calib_train_conf[k] 
                '''
            calib_conf['num_classes'] = conf_cp.data_conf.num_classes

        d2 = copy.deepcopy(d)

        f = False 
        ks = list(d2.keys()) 
        for k in ks:
            s = d2[k]
            if(k in ['calib_conf', 'training_conf'] and s is not None):        
                #k2 = sorted(s.keys())
                #for k2 in s.keys():
                #    d2[f'{k}_{k2}'] = s[k2] #'/'.join([f'{k}__{s[k]}' for k in k2])
                d2[k] = d2[k]['fpath_key']
                #f = True 
        #if f:
        #    d2.pop('calib_conf')
        
        # Checkpoint path depends on training conf only.
        z = ""
        for k in path_keys:
            z = os.path.join(z,f'{k}__{d2[k]}' )

        z2 = ""  # the key when calib_conf is None and rest params are same.
        for k in path_keys:
            if(k!= 'calib_conf'):
                z2 = os.path.join(z2,f'{k}__{d2[k]}' )
            else:
                z2 = os.path.join(z2, f'{k}__{None}')

        
        conf_cp['run_dir'] = os.path.join(conf['output_root'], z)

        conf_cp['log_file_path']  = os.path.join(conf_cp['run_dir'], conf_cp['method']+".log" )
        conf_cp['out_file_path']  = os.path.join(conf_cp['run_dir'], conf_cp['method']+".pkl" )
        conf_cp['conf_file_path'] = os.path.join(conf_cp['run_dir'],  "run_config.yaml" )


        root_dir = conf_cp['root_dir']
        ckpt_root = os.path.join(root_dir, "ckpt" )

        #ckpt_file_name =  get_model_ckpt_file_name(conf_cp)
        conf_cp.training_conf['save_ckpt'] = True 
        conf_cp.training_conf['train_from_scratch'] = True  
        conf_cp.training_conf['ckpt_save_path'] =  os.path.join(ckpt_root, conf_cp['root_pfx'], z, "model.ckpt")
        conf_cp.training_conf['ckpt_load_path'] =  os.path.join(ckpt_root, conf_cp['root_pfx'], z, "model.ckpt")

        conf_cp['key'] = z 
        conf_cp['key_no_calib'] = z2 

        conf_cp.training_conf['no_calib_ckpt_save_path'] =  os.path.join(ckpt_root, conf_cp['root_pfx'], z2, "model.ckpt")
        conf_cp.training_conf['no_calib_ckpt_load_path'] =  os.path.join(ckpt_root, conf_cp['root_pfx'], z2, "model.ckpt")

        #if(conf_cp['method']=='passive_learning' and calib_conf['type']=='post_hoc'):
            # in passive runs i.e. one round runs we can use same model checkpoint for all post-hoc methods
            # to avoid training models multiple times.
        
        if(conf_cp['calib_conf']):
            calib_conf = conf_cp['calib_conf']
            if(calib_conf['name']=='auto_label_opt_v0'):
                
                calib_conf['training_conf_t'] = copy.deepcopy(calib_conf['training_conf_g'])
                calib_conf['num_classes'] = conf_cp.data_conf['num_classes']

                #unpack model conf
                model_conf ={"input_dimension":-1, "output_dimension":calib_conf['num_classes']}
                m_conf = calib_conf['model_conf']
                # two_layer:u:a , first position is name, second is the factor i.e. second layer size will be u times the input dimension, a is the activation function.
                # e.g. two_layer:2:relu
                if(":" in m_conf):
                    z = [z.strip() for z in m_conf.split(":")]
                    if(z[0]=="two_layer"):
                        #m_conf_2 = OmegaConf.load('{}/model_confs/two_layer_net_base_conf.yaml'.format(conf_dir))
                        model_conf["layers"] = []
                        model_conf["layers"].append({"type": "linear", "dim_factor": float(z[1])})
                        model_conf["layers"].append({"type": "activation", "act_fun": z[2]})


                calib_conf['model_conf'] = model_conf 

                

        lst_confs.append(conf_cp)
    return lst_confs 


