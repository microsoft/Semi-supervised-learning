# API

## Algorithms

---


```python
class AlgorithmBase(args, net_builder, num_classes,ema_m, lambda_u, num_eval_iter, tb_log, logger, **kwargs)
```

The base class that can be **inherited** for each SSL algorithm.

```Python
__init__(self, **kwargs)
```

>The initial function.

```python
set_data_loader(self, loader_dict)
```
>Used to set data loader.

>>Parameters: 

* **loader_dict**(*dict*): The dictionary of data loaders.

```python
set_optimizer(self, optimizer, scheduler=None) 
```
>Used to set optimizer and scheduler.

>>Parameters:
- **optimizer**(*optimizer*): The optimizer.
- **scheduler**(*scheduler*): The scheduler.


```python
process_batch(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s) 
```
>Send data to GPU.

>>Paramters:
- **idx_lb**(*tensor*): The index of labeled data.
- **x_lb**(*tensor*): labeled data stored in CPU.
- **y_lb**(*tensor*): Labels of labeled data stored in CPU.
- **idx_ulb**(*tensor*): The index of unlabeled data.
- **x_ulb_w**(*tensor*): The weakly augmented unlabeled data stored in CPU.
- **x_ulb_s**(*tensor*): The strongly augmented unlabeled data stored in CPU.

>>Returns:
- **idx_lb**(*tensor*): The index of labeled data.
- **x_lb**(*tensor*): labeled data stored in GPU.
- **y_lb**(*tensor*): Labels of labeled data stored in GPU.
- **idx_ulb**(*tensor*): The index of unlabeled data.
- **x_ulb_w**(*tensor*): The weakly augmented unlabeled data stored in GPU.
- **x_ulb_s**(*tensor*): The stronly augmented unlabeled datastored in GPU.

```python
train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s) 
```
>Implement train step for each algorithm, compute loss, update model, and record logs.

>>Paramters:
- **idx_lb**(*tensor*): The index of labeled data.
- **x_lb**(*tensor*): labeled data stored in GPU.
- **y_lb**(*tensor*): Labels of labeled data stored in GPU.
- **idx_ulb**(*tensor*): The index of unlabeled data.
- **x_ulb_w**(*tensor*): The weakly augmented unlabeled data stored in GPU.
- **x_ulb_s**(*tensor*): The stronly augmented unlabeled datastored in GPU.

>>Returns:
- **tb_dic**: training logs.

```python
train(self) 
```
> Control the training epochs and iterations.
```python
before_train_step(self) 
```
> Preprocess data.
```python
after_train_step(self) 
```
> Determine whether to evaluate/save the model.
```python
evaluate(self, eval_loader) 
```
>The evaluation function.

>>Parameters:
- **eval_loader**: The data loader used for evaluation.
>>Returns: 
- **log**: Evaluation logs.

```python
save_model(save_name, save_path)
```
>Save the models for evaluation or resume.

>>Parameters:
- **save_name**(*str*): The file name used for saving.
- **save_path**(*str*): The path used for saving.

```python
load_model(load_path)
```
> Load the saved models.

>>Parameters:
- **load_path**(*str*)): The path of saved models.

```python
get_argument() 
```
> Get the algorithm arugments.

---
---


## Datasets 

Basic Utils

```python
split_ssl_data(args, data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True)
```
> Data  and target are splitted into labeled and unlabeld data.
    
>> Parameters:
- **index**(*nd.array*): If np.array of index is given, select the data[index], target[index] as labeled samples.
- **include_lb_to_ulb**(*bool*): True if labeled data is also included in unlabeld data.
>>Return:
- **lb_data**(*nd.array*): labeled data
- **lbs**(*nd.arrays*): ground-truth labels of labeled data
- **ulb_data**(*nd.array*): returned unlabeled data if *include_lb_to_ulb* is False.
- **ulbs**(*nd.array*): returned ground-truth labels of unlabeled data is *include_lb_to_ulb* is False.
- **data**(*nd.array*): returned whole data set if *include_lb_to_ulb* is True.
- **targets**(*nd.array*): returned whole ground-truth label set if *include_lb_to_ulb* is True.

---

### CV Datasets
```python
class BasicDataset(Dataset)
```
    
BasicDataset returns a pair of image and labels (targets).
If targets are not given, BasicDataset returns None as the label.
This class supports strong augmentation for Fixmatch,
and return both weakly and strongly augmented images.

```python
__init__(self, alg, data, targets=None, num_classes=None,transform=None, is_ulb=False, strong_transform=None onehot=False, *args, **kwargs)
```
 >Intiailization.
>>Parameters:
- **alg**(*str*): name of the algorithm
- **data**(*list*): x_data
- **targets**(*list, optional*): y_data
- **num_classes**(*int*): number of label classes
- **transform**(*transform*): basic transformation of data
- **use_strong_transform**(*bool*): If True, this dataset returns both weakly and strongly augmented images.
- **strong_transform**(*bool*): list of transformation functions for strong augmentation
- **onehot**(*bool*): If True, label is converted into onehot vector.

        
```python
__sample__(self, idx)
```
>Get the data of a specific index
>> Parameters:
- **idx**(*int*): the index of the data
>> Return:
- **img**: the data of the index
- **target**: the one-hot label of the index

```python
__getitem__(self, idx)
```
> If strong augmentation is not used, the weak augmented image would be returned, else weak and strong augmented image would both be returned. 
>> Parameters:
- **idx**(*int*): the index of data
>>Return:
- **weak_augment_image**: data with weak augmentation.
- **strong_augment_image**: data with strong augmentation, returned is strong augmentation is used. 
- **target**: label of indexed data. 

```python
 __len__(self)
 ```
 > Get the length of the data
 >>Returns:
 - **length**(*int*): the length of data

-----
#### **CIFAR**
Both CIFAR-10 and CIFAR-100 are supported. 

```python
get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data')
```
>Get the CIFAR-10/CIFAR-100 Dataset for semi-supervised learning
>>Parameters:
- **args**(*arg*): arguments
- **alg**(*str*): name of the algorithm
- **name**(*str*): choose from cifar10 or cifar100
- **num_labels**(*int*): number of labels
- **num_classes**(*int*): number of classes
- **data_dir**(*str, optional*): path to save data
>>Return:
- **lb_dset**: labeled training set
- **ulb_dset**: unlabeled training set
- **eval_dset**: evaluation set

---

#### **EuroSat**
EuroSat has 27000 images for 10 classes 'AnnualCrop', 'Forest', 'HerbaceousVegetation',
'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'.

RGB Sentinel-2 satellite images are used in USB.

[Original paper](https://arxiv.org/pdf/1709.00029.pdf).

Each image is of size 64x64x3. 80/20 split between training and test images is used and there is no val set. To ensure there is only a small imbalance among classes, the split is applied classwisely.

```python
get_eurosat(args, alg, dataset, num_labels, num_classes, data_dir='./data', seed=1)
```
>>Parameters:
- **args**(*arg*): arguments
- **alg**(*str*): name of the algorithm
- **num_labels**(*int*): number of labels
- **num_classes**(*int*): number of classes
- **data_dir**(*str, optional*): path to save data
- **seed**(*int*): random seed
>>Return:
- **lb_dset**: labeled training set
- **ulb_dset**: unlabeled training set
- **eval_dset**: evaluation set

---

#### **STL10**
```python
get_stl10(args, alg, name, num_labels, num_classes, data_dir='./data')
```
>>Parameters:
- **args**(*arg*): arguments
- **alg**(*str*): name of the algorithm
- **name**(*str*): stl10
- **num_labels**(*int*): number of labels
- **num_classes**(*int*): number of classes
- **data_dir**(*str, optional*): path to save data
>>Return:
- **lb_dset**: labeled training set
- **ulb_dset**: unlabeled training set
- **eval_dset**: evaluation set

---
#### **SVHN**
```python
get_svhn(args, alg, name, num_labels, num_classes, data_dir='./data')
```
>>Parameters:
- **args**(*arg*): arguments
- **alg**(*str*): name of the algorithm
- **name**(*str*): svhn
- **num_labels**(*int*): number of labels
- **num_classes**(*int*): number of classes
- **data_dir**(*str, optional*): path to save data
>>Return:
- **lb_dset**: labeled training set
- **ulb_dset**: unlabeled training set
- **eval_dset**: evaluation set

---
#### **MedMNIST**

The MedMNIST includes a set of medical datasets, namely: *pathmnist*, *octmnist*, *pneumoniamnist*, *chestmnist*, *dermamnist*, *retinamnist*, *breastmnist*, *bloodmnist*, *tissuemnist*, *organamnist*, *organcmnist*, *organsmnist*, *organmnist3d*, *adrenalmnist3d*, *fracturemnist3d*, *vesselmnist3d*, *synapsemnist3d*.

```python
get_medmnist(args, alg, dataset_name, num_labels, num_classes, data_dir='./data', seed=1):
    data_dir = os.path.join(data_dir, 'medmnist', dataset_name.lower())
```
>>Parameters:
- **args**(*arg*): arguments
- **alg**(*str*): name of the algorithm
- **dataset_name**(*str*): assignate the dataset name
- **num_labels**(*int*): number of labels
- **num_classes**(*int*): number of classes
- **data_dir**(*str, optional*): path to save data
>>Return:
- **lb_dset**: labeled training set
- **ulb_dset**: unlabeled training set
- **eval_dset**: evaluation set

---
#### **ImageNet**

```python
get_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data')
```
- **args**(*arg*): arguments
- **alg**(*str*): name of the algorithm
- **name**(*str*): assignate the dataset name
- **num_labels**(*int*): number of labels
- **num_classes**(*int*): number of classes
- **data_dir**(*str, optional*): path to save data
>>Return:
- **lb_dset**: labeled training set
- **ulb_dset**: unlabeled training set
- **eval_dset**: evaluation set

---

### NLP Dataset

```python
class BasicDataset(Dataset)
```

BasicDataset returns a pair of image and labels (targets).
If targets are not given, BasicDataset returns None as the label.
This class supports strong augmentation for Fixmatch,
and return both weakly and strongly augmented images.

```python
__init__(self, alg, data, targets=None, num_classes=None,
is_ulb=False, onehot=False, *args, **kwargs)
```
>Initialization. 
>>Parameters:
- **data**: x_data
- **targets**: y_data (if not exist, None)
- **num_classes**(*int*): number of label classes
- **is_ulb**(*int*): True is the dataset is unlabeled.
- **onehot**(*bool*): True if label is converted into onehot vector.
        
```python
random_choose_sen(self)
```
>
>>Return:
- **sen**(*int**): return 1 or 2

```python    
__getitem__(self, idx)
```
>Get data of index
>>Parameters:
- **idx**(*int*): index of data
>>Return:
- **weak_augment_image**: data with weak augmentation.
- **strong_augment_image**: data with strong augmentation, returned is strong augmentation is used. 
- **target**: label of indexed data. 

```python
 __len__(self)
 ```
 > Get the length of the data
 >>Returns:
 - **length**(*int*): the length of data

---

### Speech Datasets

```python
class BasicDataset(Dataset)
```
> BasicDataset returns a pair of image and labels (targets).
If targets are not given, BasicDataset returns None as the label. This class supports strong augmentation for Fixmatch,
and return both weakly and strongly augmented images.
```python
__init__(self, alg, data, targets=None, num_classes=None, is_ulb=False, onehot=False, max_length_seconds=15, sample_rate=16000, is_train=True, *args, **kwargs):
```

>Initialization. 
>>Parameters:
- **data**: x_data
- **targets**: y_data (if not exist, None)
- **num_classes**(*int*): number of label classes.
- **ls_unlabeled**(*bool*): True if the dataset is without labels
- **onehot**(*bool*): True if label is converted into onehot vector.
- **max_length_seconds**(*int, optional*): max length of speech using seconds.
- **sample_rate**(*int*): sampling rate.
- **is_train**(*bool*): True if dataset is used for training.

        
```python    
__getitem__(self, idx)
```
>Get data of index
>>Parameters:
- **idx**(*int*): index of data
>>Return:
- **weak_augment_image**: data with weak augmentation.
- **strong_augment_image**: data with strong augmentation, returned is strong augmentation is used. 
- **target**: label of indexed data. 

```python
 __len__(self)
 ```
 > Get the length of the data
 >>Returns:
 - **length**(*int*): the length of data

---
---

## Model Zoo

### **Bert**

```python
class ClassificationBert(nn.Module)
```
Pre-trained Bert model will be loaded and further used in training. 

```python
__init__(self, name, num_classes=2)
```
> Intilization.
>> Parameters:
- **name**(*str*): bert
- **num_classes**(*int*): number of classes

```python
 forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs)
 ```
 > Forward function. 
 >> Parameters: 
 - **x**(*tensor*): input, feature berfore the classifier if *only_fc* is *True*; 
- **only_fc**(*bool*):True if input is features before classifier.
- **only_feat**(*bool*): True if only return pooled features
- **return_embed**(*bool*): True if return word embedding, used for vat

>>Return:
- **embed**(*tensor*): word embedding if *return_embed* is *True*.
- **logits**(*tensor*): output of the classifier

```python
extract(self, x):
```
> Extract the pooled features.
>>Parameters:
- **x**(*tensor*): data

---

### **Hubert**

```python
class ClassificationHubert(nn.Module)
```
```python
 __init__(self, name, num_classes=2)
 ```
> Intilization.
>> Parameters:
- **name**(*str*): bert
- **num_classes**(*int*): number of classes

```python
forward(self, x, only_fc=False, only_feat=False, **kwargs)
```
> Forward function. 
 >> Parameters: 
 - **x**(*tensor*): input, feature berfore the classifier if *only_fc* is *True*; 
- **only_fc**(*bool*):True if input is features before classifier.
- **only_feat**(*bool*): True if only return pooled features

>>Return:
- **pooled_output**(*tensor*): pooled outputs if *only_feat* is *True*.
- **logits**(*tensor*): output of the classifier

```python
extract(self, x):
```
> Extract the pooled features.
>>Parameters:
- **x**(*tensor*): data
---

### **VisionTransformer**
```python
class VisionTransformer(nn.Module)
```
A PyTorch implementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
```python
_init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True ,drop_rate=0., attn_drop_rate=0., drop_path_rate=0., init_values=None,embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block)
```
> Initialization.
>>Parameters:
- **img_size**(*int, tuple*): input image size
- **patch_size**(*int, tuple*): patch size
- **in_chans**(*int*): number of input channels
- **num_classes**(*int*): number of classes for classification head
- **global_pool**(*str*): type of global pooling for final sequence (default: 'token')
- **embed_dim**(*int*): embedding dimension
- **depth**(*int*): depth of transformer
- **num_heads**(*int*): number of attention heads
- **mlp_ratio**(*int*): ratio of mlp hidden dim to embedding dim
- **qkv_bias**(*bool*): enable bias for qkv if True
- **representation_size**(*Optional[int]*): enable and set representation layer (pre-logits) to this value if set
- **drop_rate**(*float*): dropout rate
- **attn_drop_rate**(*float*): attention dropout rate
- **drop_path_rate**(*float*): stochastic depth rate
- **weight_init**(*str*): weight init scheme
- **init_values**(*float*): layer-scale init values
- **embed_layer**(*nn.Module*): patch embedding layer
- **norm_layer**(*nn.Module*): normalization layer
- **act_layer**(*nn.Module*): MLP activation layer

```python
forward(self, x, only_fc=False, only_feat=False, **kwargs)
```
> Forward function. 
 >> Parameters: 
 - **x**(*tensor*): input, feature berfore the classifier if *only_fc* is *True*; 
- **only_fc**(*bool*):True if input is features before classifier.
- **only_feat**(*bool*): True if only return pooled features

>>Return:
- **x**(*tensor*): feature if *only_feat* is *True*.
- **output**(*tensor*): output

```python
extract(self, x):
```
> Extract the pooled features.
>>Parameters:
- **x**(*tensor*): data

---

### **Wave2Vecv2**

```python
class ClassificationWave2Vec(nn.Module)
```
```python
 __init__(self, name, num_classes=2)
 ```
> Intilization.
>> Parameters:
- **name**(*str*): bert
- **num_classes**(*int*): number of classes

```python
forward(self, x, only_fc=False, only_feat=False, **kwargs)
```
> Forward function. 
 >> Parameters: 
 - **x**(*tensor*): input, feature berfore the classifier if *only_fc* is *True*; 
- **only_fc**(*bool*):True if input is features before classifier.
- **only_feat**(*bool*): True if only return pooled features

>>Return:
- **pooled_output**(*tensor*): pooled outputs if *only_feat* is *True*.
- **logits**(*tensor*): output of the classifier

```python
extract(self, x):
```
> Extract the pooled features.
>>Parameters:
- **x**(*tensor*): data

---

### **ResNet**

Codes are from [Pytorch Implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

---

### **WideResNet**
```python
class WideResNet(nn.Module)
```
```python
__init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, **kwargs)
```
>Initialization.
>>Parameters:
- **first_stride**(*int*): value of stride in the first block
- **num_classes**(*int*): number of classes
- **depth**(*int, optional*): depth of the network, default=28
- **widen_factor**(*int, optional*): default=2.
- **drop_rate**(*float, optional*): default=0.0

```python
forward(self, x, only_fc=False, only_feat=False, **kwargs)
```
> Forward function. 
 >> Parameters: 
 - **x**(*tensor*): input, feature berfore the classifier if *only_fc* is *True*; 
- **only_fc**(*bool*):True if input is features before classifier.
- **only_feat**(*bool*): True if only return pooled features

>>Return:
- **out**(*tensor*): pooled outputs if *only_feat* is *True*.
- **output**(*tensor*): output of the fully-connected later.
---


## Lighting
```python
def get_config(config)
```
> This function sets all configurations spanning from: Saving & loading of the model; Training configurations； Optimizer configurations; Backbone Net Configurations; Algorithms Configurations; Data Configurations; multi-GPUs & Distrbitued Training.

```python
class Trainer()
```
> This class enables training, evaluation and prediction on specific dataset.

```python
__init__(self, config, algorithm, verbose=0)
```
> Initialization.
>>Parameters:
- **config**(*config*): The configuration arguments. 
- **algorithm**(*nn.Module*): the algorithm model

```python
fit(self, train_lb_loader, train_ulb_loader, eval_loader)
```
> Conduct the entire training.
>>Parameters:
- **train_lb_loader**(*nn.dataloader*): data loader of labeled training data
- **train_ulb_loader**(*nn.dataloader*): data loader of unlabeled training data
- **eval_loader**(*nn.dataloader*): data loader of evaluation data set

```python
evaluate(self, data_loader, use_ema_model=False)
```
> Evaluate the performance of trained model on specific data.
>>Parameters:
- **data_loader**(*nn.dataloader*): data loader of testing set
- **use_ema_model**(*bool, optional*): True if use ema.

```python
predict(self, data_loader, use_ema_model=False， return_gt=False)
```
> Get the predicted class labels of testing set.
>>Paramters:
- **data_loader**(*nn.dataloader*): data loader of testing set.
- **use_ema_model**(*bool*): True if use ema
- **return_gt**(*bool*): True if return ground-truth labels
>>Return:
- **y_pred**(*tensor*): predicted labels
- **y_logits**(*tensor*): predicted logits
- **y_true**(*tensor*): ground-truth labels. Returned if *return_gt* is *True*. 
---







