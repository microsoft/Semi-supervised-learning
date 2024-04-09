import yaml 
import re
import numpy as np 
import torch
import json 


def set_seed(seed):
    import random 
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

def read_json_file(file_path):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
        return json_data

def load_yaml_config(file_path):
    al_conf = None 
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    try:
        al_conf = yaml.load(open(file_path, "r").read(), Loader=loader)
    except yaml.YAMLError as exc:
        print(exc)
    
    return al_conf 
