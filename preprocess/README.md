# Data Preprocess in USB

 This document provides the instructions for downloading and processing the datasets used in USB. Part of the datasets used in USB are allowed for re-distribution, and we provide download link directly for processed datasets of this part. The remaining datasets need to be downloaded from the original website and use the process code provided to be converted into the format used in USB.

## Download Datasets

Most of the datasets used in USB can be download and used directly:

```bash
cd ../
mkdir data & cd data
wget https://wjdcloud.blob.core.windows.net/dataset/usbdata.tar.gz
tar -xvf usbdata.tar.gz
```

The tar.gz file contains:

* CV datasets: CIFAR-10, CIFAR-100, STL-10, SVHN, EuroSAT, TissueMNIST
* NLP: Amazon Review, Yahoo Answers, Yelp Review, AG News
* Audio: FSDNoisy18k, GTZAN, ESC50

You can now directly use these datasets by setting the *data_dir* argument in configuration files as "./data/"

The data structure should be like:


```text
Semi-supervised-learning
├── semilearn
├── configs
├── train.py
├── data
│   ├── cifar10
│   │   ├── cifar-10-batches-py
│   ├── cifar100
│   │   ├── cifar-100-python
│   ├── stl10
│   │   ├── stl10_binary
│   ├── svhn
│   │   ├── train_32x32.mat
│   │   ├── test_32x32.mat
│   │   ├── extra_32x32.mat
│   ├── eurosat
│   │   ├── AnnualCrop
│   │   ├── Forest
│   │   ├── .....
│   ├── medmnist
│   │   ├── tissuemnist
│   ├── amazon_review
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   ├── ag_news
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   ├── yahoo_answers
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   ├── yelp_review
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   ├── fsdnoisy
│   │   ├── train.pkl
│   │   ├── dev.pkl
│   │   ├── test.pkl
│   ├── gtzan
│   │   ├── train.pkl
│   │   ├── dev.pkl
│   │   ├── test.pkl
│   ├── esc50
│   │   ├── train.pkl
│   │   ├── dev.pkl
│   │   ├── test.pkl
```




## Process Raw Datasets

For the remaining part of the datasets, you need to download the raw data and process them using the provided scripts. 

### Semi-Aves

Download the raw data from "https://github.com/cvl-umass/semi-inat-2020#data-and-annotations"  and "https://github.com/cvl-umass/ssl-evaluation/tree/main/data" into "./data/semi_fgvc"

Make sure the semi_fgvc folder in data follows:


```text
├── data
│   ├── semi_fgvc
│   │   ├── annotation
│   │   ├── trainval_images
│   │   ├── test
│   │   ├── u_train_in
│   │   ├── u_train_out
│   │   ├── cub
│   │   ├── semi_aves
```


### AclIMDB


Download the raw dataset from "https://ai.stanford.edu/~amaas/data/sentiment/"

Run
```bash
python preprocess/preprocess_aclimdb.py
```

Check the processed data follows:
```text
├── data
│   ├── aclImdb
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
```



### UrbanSound8k

Download the raw dataset from "https://urbansounddataset.weebly.com/urbansound8k.html"

Run
```bash
python preprocess/preprocess_urbansound.py
```

Check the processed data follows:
```text
├── data
│   ├── urbansound8k
│   │   ├── train.pkl
│   │   ├── dev.pkl
│   │   ├── test.pkl
│   │   ├── id2label.json
│   │   ├── info.txt
```


### SuperbKS


Run
```bash
python preprocess/preprocess_superb.py
```
