# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#sphx-glr-beginner-text-sentiment-ngrams-tutorial-py

import numpy as np
import torch.nn as nn

from ..dataset_utils import *

from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from torchtext.datasets import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformers import BertTokenizer, logging
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from tqdm import tqdm
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextCustom(Dataset):
    def __init__(self,data_conf,name):
        self.criterion = nn.CrossEntropyLoss()
        self.data_conf = data_conf
        self.dataset_name = name
        self.d = self.data_conf["dimension"] # this is the dimension of of embeddings
        self.num_classes = self.data_conf["num_classes"]

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
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
            # X = torch.stack([self.X[i] for i in idcs])
            X = [self.X[i] for i in idcs]
        if(self.Y is not None):
            # Y = torch.stack([self.Y[i] for i in idcs])
            Y = [self.Y[i] for i in idcs]
            return CustomTensorDataset(X=X,Y=Y,num_classes = self.num_classes, d=self.d,transform=self.transform)

    def build_dataset(self):
        
        data_conf = self.data_conf

        data_dir  = self.data_conf['data_path']
        
        if self.dataset_name == "multi_nli":
            # !wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip # download dataset
            file_path = f"{data_conf['data_path']}/tokenized_textual_entailments.json"
            
            # If tokenized text exists, load it and return
            if not data_conf['tokenize_text']:
                # Load saved tokenized text
                print('Reading data from json file...')
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                # Store it to self.X and self.Y
                self.X = [{'input_ids': np.array(data['input_ids']), 'token_type_ids': np.array(data['token_type_ids']), 'attention_mask': np.array(data['attention_mask'])} for data in data['X']]
                self.Y = data['Y']
                # Store it to self.X_test and self.Y_test
                self.X_test = [{'input_ids': np.array(data['input_ids']), 'token_type_ids': np.array(data['token_type_ids']), 'attention_mask': np.array(data['attention_mask'])} for data in data['X_test']]
                self.Y_test = data['Y_test']
                del data
                return

            # Read text multi-nli into dataframes
            self.df_train = pd.read_json(data_dir + "multinli_1.0_train.jsonl", lines= True)
            self.train = {} # Expecting 392702 data points
            self.train['gold_label'] = self.df_train['gold_label'].to_list()
            self.train['sentence1'] = self.df_train['sentence1'].to_list()
            self.train['sentence2'] = self.df_train['sentence2'].to_list()
            del self.df_train

            self.df_test = pd.read_json(data_dir + "multinli_1.0_dev_mismatched.jsonl", lines = True) # no resemblance to training data 
            self.df_test = self.df_test[self.df_test['gold_label'] != '-'] # remove rows with no gold label
            self.test = {} # Expecting 9832 data points
            self.test['gold_label'] = self.df_test['gold_label'].to_list()
            self.test['sentence1'] = self.df_test['sentence1'].to_list()
            self.test['sentence2'] = self.df_test['sentence2'].to_list()
            del self.df_test

            logging.set_verbosity_error() # To muffle warning: "Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy. So the returned list will always be empty even if some tokens have been removed."
            def tokenize_and_token_type_id(sentence1, sentence2):
                #https://github.com/huggingface/transformers/blob/bffac926ca6bc6c965a92bfbfd00c567a2c0fb90/src/transformers/tokenization_utils_base.py#L2529
                op = self.tokenizer(
                    sentence1,
                    sentence2,
                    add_special_tokens=True,
                    padding= 'max_length',
                    max_length=128, 
                    truncation='longest_first',
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='np' # 'np' for numpy
                    )
                # Returned is for returning to the model
                returned =  {'input_ids': op['input_ids'].squeeze(), 'token_type_ids': op['token_type_ids'].squeeze(), 'attention_mask': op['attention_mask'].squeeze()}
                # Dumped is for saving to json file - numpy array cannot be saved to json file
                dumped = {'input_ids': op['input_ids'].squeeze().tolist(), 'token_type_ids': op['token_type_ids'].squeeze().tolist(), 'attention_mask': op['attention_mask'].squeeze().tolist()}
                
                return returned, dumped


            label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

            print('Tokenizing training data...')
            self.X, X_dumped = [], []
            for s1, s2 in tqdm(zip(self.train['sentence1'], self.train['sentence2']), total = len(self.train['sentence1'])):
                returned, dumped = tokenize_and_token_type_id(s1,s2)
                self.X.append(returned)
                X_dumped.append(dumped)

            self.Y = [label_dict[l] for l in self.train['gold_label']]

            print('Tokenizing test data...')
            self.X_test, X_test_dumped = [], []
            for s1, s2 in tqdm(zip(self.test['sentence1'], self.test['sentence2']), total = len(self.test['sentence1'])):
                returned, dumped = tokenize_and_token_type_id(s1,s2)
                self.X_test.append(returned)
                X_test_dumped.append(dumped)

            self.Y_test = [label_dict[l] for l in self.test['gold_label']]

            # Save these values in a json file
            stored_tokens = {
                'X': X_dumped,
                'Y': self.Y,
                'X_test': X_test_dumped,
                'Y_test': self.Y_test
            }
            
            # Dump tokenized textual entailments into json file 
            with open(file_path, 'w') as json_file:
                json.dump(stored_tokens, json_file)
        else:
            raise ValueError("Text Dataset not supported")

    def get_test_datasets(self):
        X_ = self.X_test
        Y_ = self.Y_test
        return CustomTensorDataset(X=X_,Y=Y_, num_classes = self.num_classes,d=self.d,transform=self.transform)
