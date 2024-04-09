import torch
import torch.nn as nn
import torch.nn.functional as F

# batch size 128; adam; learning rate 0.001;

class CifarSmallNet(nn.Module):
    def __init__(self, n_classes=10):
        super(CifarSmallNet,self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()
    
    def get_embedding(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch, embeddings
        x = F.relu(self.fc1(x))
        embeddings = F.relu(self.fc2(x))

        out = {}
        out['embedding'] = embeddings
        return out

    def forward(self, x):
    
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch, embeddings
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pre_logits = x 

        logits = self.fc3(x)
        probs = F.softmax(logits, dim=1)
        out = {}
        out['pre_logits'] = pre_logits
        out['logits'] = logits
        out['abs_logits'] =  torch.abs(logits)
        out['probs'] = probs 
        return out