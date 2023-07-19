# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import BertTokenizer, BertTokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data import default_data_collator


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        w_features = []
        s_features_ = []
        s_features = []
        for f in features:
            f_ = {k:v for k,v in f.items() if 'text' not in k}
            input_ids = self.tokenizer(f['text'], max_length=self.max_length, truncation=True, padding=False)['input_ids']
            f_['input_ids'] = input_ids 
            w_features.append(f_)

            if 'text_s' in f:
                input_ids_s = self.tokenizer(f['text_s'], max_length=self.max_length, truncation=True, padding=False)['input_ids']
                s_features.append({'input_ids':input_ids_s})

            if 'text_s_' in f:
                input_ids_s_ = self.tokenizer(f['text_s_'], max_length=self.max_length, truncation=True, padding=False)['input_ids']
                s_features_.append({'input_ids':input_ids_s_})

        batch = self.tokenizer.pad(
            w_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        if 'label' in batch:
            if len(s_features)==0:
                return {'idx_lb': batch['idx'], 'x_lb': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, 'y_lb': batch['label']}
            else:
                s_batch = self.tokenizer.pad(
                    s_features,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                return {'idx_lb': batch['idx'], 'x_lb': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, 
                                                'x_lb_s': {'input_ids': s_batch['input_ids'], 'attention_mask': s_batch['attention_mask']}, 
                                                'y_lb': batch['label']}
        else:
            if len(s_features) > 0:
                s_batch = self.tokenizer.pad(
                    s_features,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                if len(s_features_) > 0:
                    s_batch_ = self.tokenizer.pad(
                        s_features_,
                        padding=True,
                        max_length=None,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                    return {'idx_ulb': batch['idx'], 
                            'x_ulb_w': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, \
                            'x_ulb_s_0': {'input_ids': s_batch['input_ids'], 'attention_mask': s_batch['attention_mask']}, \
                            'x_ulb_s_1': {'input_ids': s_batch_['input_ids'], 'attention_mask': s_batch_['attention_mask']}
                        }
                else:
                    return {'idx_ulb': batch['idx'], 'x_ulb_w': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, 'x_ulb_s': {'input_ids': s_batch['input_ids'], 'attention_mask': s_batch['attention_mask']}}
            else:
                return {'idx_ulb': batch['idx'], 'x_ulb_w': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}}


def get_bert_base_uncased_collactor(max_length=512):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    collact_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
    return collact_fn


def get_bert_base_cased_collactor(max_length=512):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    collact_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
    return collact_fn