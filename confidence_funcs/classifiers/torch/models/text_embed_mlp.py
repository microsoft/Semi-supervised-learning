# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#sphx-glr-beginner-text-sentiment-ngrams-tutorial-py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS

class TextClassificationModel(nn.Module):

    def __init__(self, model_conf):
        super(TextClassificationModel, self).__init__()
        vocab_size, embed_dim, num_class = model_conf['vocab_size'], model_conf['embed_dim'], model_conf['num_classes']
        self.device = model_conf['device']
        self.criterion = nn.CrossEntropyLoss()
        
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

        self.tokenizer = get_tokenizer('basic_english')
        # self.train_iter = AG_NEWS(split='train',root=model_conf['data_path'])
        # self.vocab = build_vocab_from_iterator(self.__yield_tokens(self.train_iter), specials=["<unk>"])
        # self.vocab.set_default_index(self.vocab["<unk>"])
        # self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        # self.label_pipeline = lambda x: int(x) - 1

    def __yield_tokens(self, data_iter):
        for _, text in data_iter:
            x = self.tokenizer(text)
            yield x 
    
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, X):

        # Tokenize and build vocab
        self.vocab = build_vocab_from_iterator(self.__yield_tokens(X), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        
        text = [torch.tensor(self.text_pipeline(x)) for x in X]
        offsets = [0] + [len(x) for x in text]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)

        offsets = offsets.to(device=self.device)
        text = text.to(device=self.device)
        
        embedded = self.embedding(text, offsets)
        logits = self.fc(embedded)
        probs = F.softmax(logits, dim=1)
        out = {}
        out['probs'] = probs 
        out['logits'] = logits
        out['abs_logits'] =  torch.abs(logits)
        return out