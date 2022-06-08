import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = load_state_dict_from_url(checkpoint_path, map_location='cpu')

    
    orig_state_dict = checkpoint['model']
    new_state_dict = {}
    for key, item in orig_state_dict.items():

        
        if key.startswith('module'):
            key = '.'.join(key.split('.')[1:])
        
        # TODO: better ways
        if key.startswith('fc') or key.startswith('classifier') or key.startswith('mlp') or key.startswith('head'):
            continue
            
        # check vit and interpolate
        # if isinstance(model, VisionTransformer) and 'patch_emb'

        if key == 'pos_embed':
            posemb_new = model.pos_embed.data
            posemb = item
            item = resize_pos_embed_vit(posemb, posemb_new)

        new_state_dict[key] = item 
    
    match = model.load_state_dict(new_state_dict, strict=False)
    print(match)
    return model



def resize_pos_embed_vit(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    import math
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    # _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb