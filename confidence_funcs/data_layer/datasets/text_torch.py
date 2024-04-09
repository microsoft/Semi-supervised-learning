# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#sphx-glr-beginner-text-sentiment-ngrams-tutorial-py

import numpy as np
import torch.nn as nn

from datasets.dataset_utils import *

from torch.utils.data import Dataset
import numpy as np

from torchtext.datasets import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torchtext.datasets import IMDB

class TextTorch(Dataset):
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
            X = tuple([self.X[i] for i in idcs] )
        if(self.Y is not None):
            Y = tuple([self.Y[i] for i in idcs] )
            return CustomTensorDataset(X=X,Y=Y,num_classes = self.num_classes, d=self.d,transform=self.transform)

    def build_dataset(self):
        data_dir  = self.data_conf['data_path']
        
        if self.dataset_name == "AG_NEWS":
            self.data  = AG_NEWS(split='train',root=data_dir)
            self.test_data  = AG_NEWS(split='test',root=data_dir)
        elif self.dataset_name == "IMDB":
            self.data =  IMDB(split='train',root=data_dir)
            self.test_data = IMDB(split='test',root=data_dir)
        else:
            raise ValueError("Text Dataset not supported")
        
        def get_XY(data):
            labels = []
            texts = []
            for (label, text) in data:
                labels.append(label)
                texts.append(text)
            return texts, torch.tensor(labels)-1 # minus one for o indexing
        
        self.reviews_train , self.Y  = get_XY(self.data)
        self.reviews_test  , self.Y_test  = get_XY(self.test_data)

        from transformers import BertTokenizer 
        PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        MAX_LEN = 512
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        X = [] 
        for review in self.reviews_train + self.reviews_test:

            enc = tokenizer.encode_plus( review, max_length=MAX_LEN, add_special_tokens=True,
                                        return_token_type_ids=False,
                                        pad_to_max_length=True,
                                        return_attention_mask=True,
                                        return_tensors='pt',truncation=True)
            X.append({'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()})

        self.X = X[:len(self.reviews_train)]
        self.X_test = X[len(self.reviews_train):]

        
    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return CustomTensorDataset(X=X_,Y=Y_, num_classes = self.num_classes,d=self.d,transform=self.transform)
