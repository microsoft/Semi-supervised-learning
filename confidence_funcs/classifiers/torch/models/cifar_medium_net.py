import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarMediumNet(nn.Module):
    def __init__(self, n_classes=10):
        super(CifarMediumNet,self).__init__()
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            )
        self.linear = nn.Linear(512, 10)

    def get_embedding(self,x):
        embeddings = self.network(x)
        
        out = {}
        out['embedding'] = embeddings
        return out
        
    def forward(self, x):

        # logits = self.linear(self.network(x))
        
        x = self.network(x)
        logits = self.linear(x)

        probs = F.softmax(logits, dim=1)
        out = {}
        out['pre_logits']= x 
        out['logits'] = logits
        out['abs_logits'] =  torch.abs(logits)
        out['probs'] = probs 
        return out