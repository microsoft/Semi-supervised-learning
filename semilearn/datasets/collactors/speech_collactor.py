# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import AutoFeatureExtractor
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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
    sample_rate: Optional[int] = 16000
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # check if input_values_s in features
        s_features_ = []
        s_features = []
        w_features = []

        for f in features:
            f_ = {k:v for k,v in f.items() if 'wav' not in k}
            input_values = self.tokenizer(f['wav'], max_length=int(self.max_length * self.sample_rate), 
                                          sampling_rate=self.sample_rate, truncation=True)['input_values'][0]
            f_['input_values'] = input_values
            w_features.append(f_)

            if 'wav_s' in f:
                input_values_s = self.tokenizer(f['wav_s'], max_length=int(self.max_length * self.sample_rate), 
                                                sampling_rate=self.sample_rate, truncation=True)['input_values'][0]
                s_features.append({'input_values':input_values_s})
            
            if 'wav_s_' in f:
                input_values_s_ = self.tokenizer(f['wav_s_'], max_length=int(self.max_length * self.sample_rate), 
                                                sampling_rate=self.sample_rate, truncation=True)['input_values'][0]
                s_features_.append({'input_values':input_values_s_})

        batch = self.tokenizer.pad(
            w_features,
            padding=self.padding,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
            
        if 'labels' in batch:
            return {'idx_lb': batch['idx'], 'x_lb': batch['input_values'], 'y_lb': batch['labels']}
        else:
            if len(s_features) > 0 and len(s_features_) == 0:
                s_batch = self.tokenizer.pad(
                    s_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                return {'idx_ulb': batch['idx'], 'x_ulb_w': batch['input_values'], 'x_ulb_s': s_batch['input_values']}
            elif len(s_features) > 0 and len(s_features_) > 0:
                s_batch = self.tokenizer.pad(
                    s_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                s_batch_ = self.tokenizer.pad(
                    s_features_,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                return {'idx_ulb': batch['idx'], 'x_ulb_w': batch['input_values'], 'x_ulb_s_0': s_batch['input_values'], 'x_ulb_s_1': s_batch_['input_values']}
            else:
                return {'idx_ulb': batch['idx'], 'x_ulb_w': batch['input_values']}


def get_wave2vecv2_base_collacor(max_length=4, sample_rate=16000):
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
    collator = DataCollatorWithPadding(feature_extractor, max_length=max_length, sample_rate=sample_rate)
    return collator


def get_hubert_base_collacor(max_length=4, sample_rate=16000):
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
    collator = DataCollatorWithPadding(feature_extractor, max_length=max_length, sample_rate=sample_rate)
    return collator