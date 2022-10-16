# This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
#  The dataset consists of 5-second-long recordings of sampleing rate 44100 organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories
#  We use fold 1, 2, 3 as training data, fold 4 as validation data, fold 5 as testing data



import os
import json
import pickle
from io import BytesIO
import pandas as pd 
import librosa
import numpy as np
from glob import glob
from tqdm import tqdm


data_dir = './raw_data/UrbanSound8K'
save_dir = './data/urbansound8k'
os.makedirs(save_dir, exist_ok=True)


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def process_data(sample_rate=16000):

    meta_data = pd.read_csv(os.path.join(data_dir, 'metadata/UrbanSound8K.csv'))
    meta_dict = {}
    id2label = {}
    meta_data = meta_data.reset_index()  # make sure indexes pair with number of rows
    for _, row in meta_data.iterrows():
        meta_dict[row["slice_file_name"]] = {'fold': row["fold"], 'label': row["classID"]}
        label_id =  row["classID"]
        label = row["class"]
        if label_id not in id2label:
            id2label[label_id] = label
    

    train_data = {}
    train_cnt = 0
    dev_data = {}
    dev_cnt = 0
    test_data = {}
    test_cnt = 0
    recording_list = sorted(glob(os.path.join(data_dir, 'audio', 'fold*', '*.wav')))
    for recording in tqdm(recording_list):
        signal, rate = librosa.load(recording, sr=sample_rate, mono=True)
        filename = recording.split('/')[-1]
        fold, label = meta_dict[filename]["fold"], meta_dict[filename]["label"]

        if fold in [1, 2, 3, 4, 5, 6, 7, 8]:
            train_data[str(train_cnt)] = {}
            train_data[str(train_cnt)]["wav"] = array_to_bytes(signal)
            train_data[str(train_cnt)]["label"] = label
            train_cnt += 1
        elif fold == 9:
            dev_data[str(dev_cnt)] = {}
            dev_data[str(dev_cnt)]["wav"] = array_to_bytes(signal)
            dev_data[str(dev_cnt)]["label"] = label
            dev_cnt += 1
        else:
            test_data[str(test_cnt)] = {}
            test_data[str(test_cnt)]["wav"] = array_to_bytes(signal)
            test_data[str(test_cnt)]["label"] = label
            test_cnt += 1

    with open(os.path.join(save_dir, 'id2label.json'), 'w') as f:
        json.dump(id2label, f,  indent=4)
    

    with open(os.path.join(save_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    

    with open(os.path.join(save_dir, 'dev.pkl'), 'wb') as f:
        pickle.dump(dev_data, f)
    
    with open(os.path.join(save_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    print("train data {}, dev data {}, test data {}".format(train_cnt, dev_cnt, test_cnt))

    with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
        f.write("max length: 4 seconds \n")
        f.write("sampling rate: 44100 Hz (resampled at 16000) \n")
        f.write(f"num classes: {len(id2label)} \n")
        f.write(f"train data: {train_cnt} \n")
        f.write(f"dev data: {dev_cnt} \n")
        f.write(f"test data: {test_cnt} \n")


if __name__ == '__main__':
    process_data()