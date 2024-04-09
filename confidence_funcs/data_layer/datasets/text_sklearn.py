# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#sphx-glr-beginner-text-sentiment-ngrams-tutorial-py

import numpy as np
import torch.nn as nn

from ..dataset_utils import *

from torch.utils.data import Dataset
import numpy as np

from torchtext.datasets import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from sklearn.datasets import fetch_20newsgroups

from sentence_transformers import SentenceTransformer, util
from pathlib import Path

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextSklearn(Dataset):
    def __init__(self,data_conf,name):
        self.criterion = nn.CrossEntropyLoss()
        self.data_conf = data_conf
        self.dataset_name = name
        self.d = self.data_conf["dimension"] # this is the dimension of of embeddings
        self.num_classes = self.data_conf["num_classes"]
        
        self.transform = None # not used in this case

    def __getitem__(self, index):
        
        x = self.X[index]
        y = self.Y[index]
        
        return x, y, index 
    
    def __len__(self):
        return len(self.X)
    
    def len(self):
        return len(self.X)
    
    def get_subset(self,idcs,Y_=None):
        X = None
        Y = None 
        if(self.X is not None):
            X = torch.stack([self.X[i] for i in idcs])
        if(self.Y is not None):
            Y = torch.stack([self.Y[i] for i in idcs])
            return CustomTensorDataset(X=X,Y=Y,num_classes = self.num_classes, d=self.d,transform=self.transform)

    def build_dataset(self):
        
        data_conf = self.data_conf

        data_dir  = self.data_conf['data_path']
        
        if self.dataset_name == "twenty_newsgroups":
            # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
            self.data  = fetch_20newsgroups(subset='train',data_home=data_dir, shuffle=False, return_X_y=True)
            self.data = [(y, x) for x, y in zip(*self.data)] # Converts ([x1,x1],[y1,y2]) to [(y1,x1),(y2,x2)] to follow pytorch dataset format
            self.test_data  = fetch_20newsgroups(subset='test',data_home=data_dir, shuffle=False, return_X_y=True)
            self.test_data = [(y, x) for x, y in zip(*self.test_data)] 
        else:
            raise ValueError("Text Dataset not supported")

        from random import shuffle
        def get_XY(data):
            labels_,labels = [], []
            texts_,texts = [], []
            for label, text in data:  
                labels_.append(label+1)# +1 so that labels start from 1
                texts_.append(text)
            index_shuf = list(range(len(labels_)))
            shuffle(index_shuf)

            for i in index_shuf:
                labels.append(labels_[i])
                texts.append(texts_[i])

            return texts, torch.tensor(labels)-1 # minus one for o indexing
        
        ckpt_path    = f"{data_conf['emb_path']}/{data_conf['name']}_{data_conf['emb_model'].replace('/','_')}.pt"
        if(data_conf.compute_emb):
            self.reviews_train , self.Y  = get_XY(self.data)
            self.reviews_test  , self.Y_test  = get_XY(self.test_data)

            model = SentenceTransformer(data_conf['emb_model'])
            
            self.X      = model.encode(self.reviews_train)
            self.X_test = model.encode(self.reviews_test)

            self.X      = torch.Tensor(self.X)
            self.X_test = torch.Tensor(self.X_test)

            ckpt_content = {'reviews_train':self.reviews_train,
                            'reviews_test':self.reviews_test  ,
                            'X':self.X, 'Y' :self.Y ,
                            'X_test' : self.X_test, 'Y_test': self.Y_test}
            
            Path(data_conf['emb_path']).mkdir(parents=True, exist_ok=True)

            torch.save(ckpt_content, ckpt_path)

        else:
            print(ckpt_path)
            ckpt_content = torch.load(ckpt_path)
            self.reviews_train = ckpt_content['reviews_train']
            self.reviews_test  = ckpt_content['reviews_test']
            self.X = ckpt_content['X']
            self.Y = ckpt_content['Y']
            self.X_test = ckpt_content['X_test']
            self.Y_test = ckpt_content['Y_test']

    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return CustomTensorDataset(X=X_,Y=Y_, num_classes = self.num_classes,d=self.d,transform=self.transform)
