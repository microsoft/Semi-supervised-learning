from random import random
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
from .datasets.torch_dataset import CustomTensorDataset
#from .custom_dataset import CustomDataset

def tensorize(np_dataset,transform=None):
    X_tensor = None 
    Y_tensor = None 
    if(np_dataset.X is not None):
        X_tensor = torch.Tensor(np_dataset.X)
    if(np_dataset.Y is not None):
        Y_tensor = torch.Tensor(np_dataset.Y).long()
    tensor_dataset = CustomTensorDataset(X_tensor,Y_tensor, transform = transform)
    
    return tensor_dataset

def randomly_split_dataset(ds,fraction=0.5,random_state=42):
    out = randomly_split_ds_idcs(ds,fraction,random_state)
    return out['ds_std_train'], out['ds_std_val']

def randomly_split_ds_idcs(ds,fraction=0.5,random_state=42, num_hyp_val=1000) :
    n = ds.len()
    idcs = list(range(n))
    ps = np.random.get_state()

    np.random.seed(random_state)
    idcs_val     = np.random.choice(idcs,int(n*fraction),replace=False)
    idcs_val_set = set(list(idcs_val))
    idcs_train   = np.array( list(set(idcs).difference(idcs_val_set) ))

    ds_std_val, ds_std_train    = ds.get_subset(idcs_val), ds.get_subset(idcs_train)

    np.random.set_state(ps)
    #print(len(idcs1),len(idcs2))
     
    return {'idcs_std_train': idcs_train,'idcs_std_val':idcs_val,'ds_std_train':ds_std_train, 'ds_std_val':ds_std_val}


def take_subset_of_train_dataset(dataset,idcs):
    subset_train_ds = dataset.train_dataset.get_subset(idcs)
    return CustomDataset(subset_train_ds,dataset.test_dataset)

def take_subset_of_train_dataset(dataset,idcs):
    subset_train_ds = dataset.train_dataset.get_subset(idcs)
    return CustomDataset(subset_train_ds,dataset.test_dataset)


def getDataLoaderForSubset(tensorDataset,subsetIndices,batchSize,true_labels=True):
    
    X_np = tensorDataset.data.numpy()[subsetIndices]
    if(true_labels):
        Y_np = tensorDataset.targets.numpy()[subsetIndices]
    else:
        Y_np = np.zeros(len(subsetIndices))

    subsetDataset = CustomTensorDataset(torch.Tensor(X_np),torch.Tensor(Y_np).long(),transform=None)
    subsetLoader  = DataLoader(dataset=subsetDataset,batch_size= batchSize, shuffle=False)
    return subsetLoader

def get_data_loader_from_numpy_arrays(X_np,Y_np,batch_size,transform,shuffle):
    subsetDataset = CustomTensorDataset(torch.Tensor(X_np),torch.Tensor(Y_np).long(),transform=transform)
    subsetLoader  = DataLoader(dataset=subsetDataset,batch_size= batch_size, shuffle=shuffle)
    return subsetLoader  
    
