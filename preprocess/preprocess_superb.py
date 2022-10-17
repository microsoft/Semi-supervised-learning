import os
import json
import pickle
from io import BytesIO
import numpy as np
from tqdm import tqdm

import datasets
from datasets import DatasetDict, load_dataset
from transformers import AutoFeatureExtractor


save_path = './data'

def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def process_data(dataset_name='superb', dataset_config_name='ks', audio_column_name='audio', label_column_name='label', data_dir=None):
    """
     Keyword Spotting subset of the SUPERB dataset.
    """
    save_dataset_name = dataset_name + dataset_config_name if dataset_config_name is not None else dataset_name
    save_dir = os.path.join(save_path, save_dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    splits = ['train', 'validation', 'test']
    raw_datasets = DatasetDict()
    for split in splits:
        if data_dir is None:
            raw_datasets[split] = load_dataset(dataset_name, dataset_config_name, split=split)
        else:
            raw_datasets[split] = load_dataset(dataset_name, dataset_config_name, split=split, data_dir=data_dir)

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        'facebook/wav2vec2-base',
        return_attention_mask=True,
        revision='main'
    )


    raw_datasets = raw_datasets.cast_column(audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate))

    # Prepare label mappings.
    labels = raw_datasets["train"].features[label_column_name].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    for split in splits:
        print("processing {} for {}...".format(split, dataset_name))
        label_cnt = {label: 0 for label in labels}
        dataset = raw_datasets[split]
        data = {}
        cnt = 0
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            # for debug
            # array =  sample["audio"]["array"]
            # bytes_array = array_to_bytes(array)
            # array_bytes = bytes_to_array(bytes_array)
            # assert (array == array_bytes).all()
            label = sample[label_column_name]
            label_name = id2label[str(label)] 
            if dataset_config_name == 'ks':
                if label_name == '_silence_' or label_name == '_unknown_':
                    continue

            array = sample[audio_column_name]["array"]
            data[str(cnt)] = {}
            data[str(cnt)]["wav"] = array_to_bytes(array)
            data[str(cnt)]["label"] = label
            label_cnt[id2label[str(sample[label_column_name])]] += 1
            # print(data[str(idx)])
            cnt += 1
        
        print("split {}, cnt: {}, label cnt: {}".format(split, cnt, label_cnt))

        with open(os.path.join(save_dir, '{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(data, f)

    if dataset_config_name == 'ks':
        id2label.pop(label2id['_silence_'])
        id2label.pop(label2id['_unknown_'])

    with open(os.path.join(save_dir, 'id2label.json'), 'w') as f:
        json.dump(id2label, f,  indent=4)


if __name__ == '__main__':
    process_data(dataset_name='superb', dataset_config_name='ks', audio_column_name='audio', label_column_name='label')