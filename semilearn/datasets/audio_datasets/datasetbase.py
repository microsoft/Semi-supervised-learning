# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import random
import torchaudio

from torch.utils.data import Dataset
from semilearn.datasets.utils import get_onehot, random_subsample



class WaveformTransforms:
    def __init__(self, sample_rate=16000, max_length=1.0, n=2):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.n = n
    
    def __call__(self, wav):
        speed = 0.5 + 1.5 * random.random()
        pitch = -2.0 + 4.0 * random.random()
        attenuation = int(-5.0 + 10.0 * random.random())

        effects_list = [
            # ["lowpass", "-1", "300"],  # apply single-pole low-pass filter
            ['gain', '-n', f'{attenuation:.5f}'],  # apply 10 db attenuation
            ["pitch", f'{pitch:.5f}'],
            ["speed", f'{speed:.5f}'],  # reduce the speed
            # This only changes sample rate, so it is necessary to
            # add `rate` effect with original sample rate after this.
            ["reverb", "-w"],  # Reverberation gives some dramatic feeling
        ]
        effects = random.choices(effects_list, k=self.n)

        effects.append(["rate", f"{self.sample_rate}"])

        wav = torch.from_numpy(wav).reshape(1, -1)
        aug_wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate=self.sample_rate, effects=effects)
        aug_wav = aug_wav.numpy()[0]
        return aug_wav



class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """
    # add transform 
    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 is_ulb=False,
                 onehot=False,
                 max_length_seconds=15,
                 sample_rate=16000,
                 is_train=True,
                 *args, 
                 **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets
        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        self.max_length = max_length_seconds
        self.sample_rate = sample_rate
        self.is_train = is_train

        # self.feature_extractor = AutoFeatureExtractor.from_pretrained(net)
        self.transform = None
        self.strong_transform = WaveformTransforms(sample_rate=sample_rate, max_length=max_length_seconds)


    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented wavs
        raw_wav = self.data[idx]
        # random sample for train
        if self.is_train:
            wav = random_subsample(raw_wav, max_length=self.max_length, sample_rate=self.sample_rate)
        else:
            wav = raw_wav

        if self.is_ulb == False:
            if self.alg == 'defixmatch':
                raw_wav_s = self.strong_transform(raw_wav)
                wav_s = random_subsample(raw_wav_s, max_length=self.max_length, sample_rate=self.sample_rate)
                return {'idx':idx, 'wav':wav, 'wav_s': wav_s, 'label':target}
            else:
                return {'idx':idx, 'wav':wav, 'label':target} 
        else:
            if self.alg == 'fullysupervised' or self.alg == 'supervised':
                return {'idx':idx, 'wav':wav, 'label':target} 
            elif self.alg == 'pseudolabel' or self.alg == 'vat':
                return {'idx':idx, 'wav':wav} 
            elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                wav_w = random_subsample(raw_wav, max_length=self.max_length, sample_rate=self.sample_rate)
                return {'idx':idx, 'wav':wav, 'wav_s':wav_w}
            elif self.alg == 'comatch' or self.alg == 'remixmatch':
                raw_wav_s = self.strong_transform(raw_wav)
                wav_s = random_subsample(raw_wav_s, max_length=self.max_length, sample_rate=self.sample_rate)

                raw_wav_s_ = self.strong_transform(raw_wav)
                wav_s_ = random_subsample(raw_wav_s_, max_length=self.max_length, sample_rate=self.sample_rate)

                return {'idx':idx, 'wav': wav, 'wav_s':wav_s, 'wav_s_':wav_s_}
            else:
                raw_wav_s = self.strong_transform(raw_wav)
                wav_s = random_subsample(raw_wav_s, max_length=self.max_length, sample_rate=self.sample_rate)
                return {'idx':idx, 'wav':wav, 'wav_s': wav_s}
    
    def __len__(self):
        return len(self.data)