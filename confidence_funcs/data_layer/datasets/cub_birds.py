from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import torch
from datasets.dataset_utils import *

class GrayScaleToRGB():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, tensor):
        if len(tensor.size()) == 2:
            tensor = tensor[..., None]
        if tensor.size()[0] == 1:
            tensor = torch.cat((tensor, tensor, tensor), dim=0)
        return tensor

class ImagePathDataset(Dataset):
    def __init__(self, data_x, data_y, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_x = data_x
        self.data_y = data_y
        self.transform = transform

    def __len__(self):
        # noinspection PyTypeChecker
        return len(self.data_x)

    def __getitem__(self, idx):
        sample_x = Image.open(str(self.data_x[idx]))
        sample_y = self.data_y[idx]
        if self.transform:
            sample_x = self.transform(sample_x)

        return (sample_x, sample_y, idx )

class CUB_BirdsData:
    def __init__(self,conf):
        self.conf = conf
        self.trainData = None
        self.testData = None
        self.train_labels = None
        self.train_instances = None
        
        
    def buildDataset(self):
        
        dataConf = self.conf['data_conf']
        dataDir  = dataConf['data_path']
        
        self.transform = None
        
        transform = transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             GrayScaleToRGB(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])
                                            ])
 
        
        with open(dataDir+'/images.txt') as img_f:
            image_paths = [str(dataDir+'/images/'+l.split(' ')[1]).strip() for l in img_f.readlines()]
            image_paths = np.array(image_paths,dtype=str)
        print(dataDir+'/train_test_split.txt')
        with open(dataDir+'/train_test_split.txt') as tr_te_sp:
            train_test_split = [int(l.split(' ')[1]) for l in tr_te_sp.readlines()]
            train_test_split = np.array(train_test_split, dtype=bool)
            
        with open(dataDir+'/image_class_labels.txt') as img_cls_lbls:
            labels = [int(l.split(' ')[1])-1 for l in img_cls_lbls.readlines()]
            labels = np.array(labels)
            
        self.n_classes = len(np.unique(labels)) 
        
        self.train_instances = []
        self.train_labels = []
        self.test_instances = []
        self.test_labels = []
        
        train_paths = []
        test_paths  = []
        train_labels = []
        test_labels = []
        print(len(train_test_split))
        
        for i,v in enumerate(train_test_split):
            
            if(v==1):
                train_paths.append(image_paths[i])
                train_labels.append(labels[i])
            else:
                test_paths.append(image_paths[i])
                test_labels.append(labels[i])
        
        if('sub_sample_fraction' in dataConf):
            np.random.seed(0)
            n = len(train_labels)
            idcs = np.array(range(n))
            th = int(n*dataConf['sub_sample_fraction'])
            np.random.shuffle(idcs)
            idcs = idcs[:th]
            for i in idcs:
                img = Image.open(train_paths[i])
                img_ = transform(img).numpy()
                img.close()
                self.train_instances.append(img_)
                self.train_labels.append(train_labels[i])
            
            self.train_labels = np.array(self.train_labels)
            print(len(self.train_instances),len(self.train_labels))
            print(type(self.train_instances[0][0][0][0]))
            print(type(self.train_labels[0]))
            
        self.trainData = SubsetTensorDataset(torch.Tensor(self.train_instances),
                                             torch.Tensor(self.train_labels).long())#,#.long(),
                                                      #transform=self.transform)
            
        n = len(test_labels)
        idcs = np.array(range(n))
        th = int(n*dataConf['sub_sample_fraction'])
        np.random.shuffle(idcs)
        idcs = idcs[:th]

        for i in idcs:
            img = Image.open(test_paths[i])
            #img_np = np.array(img)
            img_ = transform(img).numpy()
            img.close()
            self.test_instances.append(img_)
            self.test_labels.append(test_labels[i])
            
        self.testData  = SubsetTensorDataset(torch.Tensor(self.test_instances),
                                             torch.Tensor(self.test_labels).long())#,#.long(),
                                             #transform=self.transform)
            
            
  
    def getTrainLoader(self,batchSize=32):
        return  DataLoader(self.trainData, batch_size=batchSize, shuffle=True, num_workers=1)
    
    def getTestLoader(self,batchSize=32):
        return  DataLoader(self.testData, batch_size=batchSize, shuffle=True, num_workers=1)
    
    def get_true_label(self,i):
        return self.train_labels[i]
    
    def get_traing_instance(self,i):
        return self.train_instances[i]

        
   
