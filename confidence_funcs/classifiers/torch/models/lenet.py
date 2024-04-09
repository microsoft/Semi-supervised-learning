import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from collections import OrderedDict


class LeNet5(nn.Module):

    def __init__(self, n_classes=10, dropout_prob=0):
        super(LeNet5, self).__init__()
        self.n_classes = n_classes
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
        )

        self.clf_layer_1 = nn.Linear(in_features=120, out_features=84)
        self.clf_layer_2 = nn.Tanh()
        self.last_drop_out = nn.Dropout(p=dropout_prob)
        self.clf_layer_3 = nn.Linear(in_features=84, out_features=n_classes)

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        #print(x.shape)
        
        x = F.pad(input=x, pad=(2, 2, 2, 2), mode='constant', value=0)
        x = x[:,None,:,:]
        x = x.float()
        #print(x.shape)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)

        x = self.clf_layer_2(self.clf_layer_1(x))
        x = self.last_drop_out(x)

        out = {}

        out['pre_logits']= x 

        #logits = self.classifier(x)
        logits = self.clf_layer_3(x)

        #logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        
        out['features'] = logits 

        out['probs'] = probs 
        out['logits'] = logits
        out['abs_logits'] =  torch.abs(logits)
        return out

    def get_embedding(self, x):
        embedding = self.feature_extractor(x)
        embedding = torch.flatten(embedding, 1)
        return embedding

    # might be problematic
    def get_grad_embedding(self,x):

        out = self.get_embedding(x)
        embDim = out.size(1)
        batchProbs = F.softmax(out, dim=1).cpu().numpy()
        out = out.cpu().numpy()
        maxInds = np.argmax(batchProbs,1)
        
        grad_embedding = np.zeros([len(x),  embDim* self.n_classes]) # matrix to store the gradient
        for j in range(len(x)):
            for c in range(self.n_classes):
                if c == maxInds[j]:
                    grad_embedding[j][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                else:
                    grad_embedding[j][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
        
        return torch.Tensor(grad_embedding)

    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad = False

    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad = True

    


class LeNet5Confidence(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5Confidence, self).__init__()
        self.n_classes = n_classes
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            #nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            #nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
            #nn.ReLU()
        )

        self.classifier = nn.Sequential(    
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),   
            #nn.ReLU(),
            #nn.Linear(in_features=84, out_features=84),
            #nn.ReLU(),
            #nn.Linear(in_features=84, out_features=42),
            #nn.ReLU(),
            nn.Linear(in_features=84, out_features=1),
            #nn.Linear(in_features=n_classes, out_features=1),
        )
        self.criterion = nn.MSELoss()


    def forward(self, x):
        #print(x.shape)
        
        x = F.pad(input=x, pad=(2, 2, 2, 2), mode='constant', value=0)
        x = x[:,None,:,:]
        x = x.float()
        #print(x.shape)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs =  logits #F.softmax(logits, dim=1)
        out = {}
        out['probs'] = probs 
        out['logits'] = logits
        out['abs_logits'] =  torch.abs(logits)
        return out