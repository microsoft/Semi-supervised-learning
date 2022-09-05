# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn
from transformers import HubertModel


class ClassificationHubert(nn.Module):
    def __init__(self, name, num_classes=2):
        super(ClassificationHubert, self).__init__()
        self.model = HubertModel.from_pretrained(name)
        # for vat
        self.model.feature_extractor._requires_grad = False 
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(*[
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, num_classes)
        ])

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            logits = self.classifier(x)
            return logits

        pooled_output = self.extract(x)

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output}
        return result_dict

    def extract(self, x):
        out_dict = self.model(x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        embed = out_dict['hidden_states'][0]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}model.feature_projection|^{}model.feature_extractor|^{}model.encoder.pos_conv_embed'.format(prefix, prefix, prefix), blocks=r'^{}model.encoder.layers.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []


def hubert_base(pretrained=False, pretrained_path=None, **kwargs):
    model = ClassificationHubert(name='facebook/hubert-base-ls960', **kwargs)
    return model


if __name__ == '__main__':
    model = hubert_base()
    print(model)
