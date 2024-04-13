import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2PreTrainedModel
import os

class CustomDINONormModel(nn.Module):
    def __init__(self, name, num_classes=8):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = Dinov2Model.from_pretrained(name)
        self.classifier = nn.Sequential(*[
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        ])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        # Extract features using DinoV2 model
        if return_embed:
            embed = self.dino_model(x)
            return embed

        out_dict = self.dino_model(x, output_hidden_states=True, return_dict=True)
        last_hidden_state = out_dict['last_hidden_state']
        pooled_output = torch.mean(last_hidden_state, 1)  # Perform mean pooling

        if only_fc:
            logits = self.classifier(pooled_output)
            return logits

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {'logits': logits, 'feat': pooled_output}
        return result_dict
        
    
    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}dino_model.embeddings'.format(prefix), blocks=r'^{}dino_model.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []



def dinov2_vitl14(pretrained=True, pretrained_path=None, **kwargs):
    model = CustomDINONormModel(name='facebookresearch/dinov2_vitl14', **kwargs)
    return model


def dinov2_vitb14(pretrained=True, pretrained_path=None, **kwargs):
    model = CustomDINONormModel(name='facebookresearch/dinov2_vitb14', **kwargs)
    return model
