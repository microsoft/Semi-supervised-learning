
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import * 

class TextClassifierMLPHead(nn.Module):

  def __init__(self, model_conf):

    super(TextClassifierMLPHead, self).__init__()
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # self.drop = nn.Dropout(p=0.3)
    self.head = nn.Linear(self.bert.config.to_dict()['hidden_size'], model_conf['num_classes'])
    self.device = model_conf['device']
    self.bert.to(self.device)
  
  def forward(self, x):
    input_ids = x["input_ids"].to(self.device)
    attention_mask = x["attention_mask"].to(self.device)
    token_type_ids = x["token_type_ids"].to(self.device)
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= token_type_ids)
    out    = self.head(output[1])
    x = out.dim()
    input_dimension = 0 if x == 0 or x == 1 or x == 3 else 1
    probs = F.softmax(out, dim=input_dimension)
    output = {}
    output['probs'] = probs 
    output['abs_logits'] =  torch.abs(out)
    output['logits'] = out 
    
    return output 