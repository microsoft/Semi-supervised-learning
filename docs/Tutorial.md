# Tutorial

This tutorial would walk through the basic usage of using USB to benchmark any provided algorithms, customize algorithms, customize dataset and the introduction of lighting version. 

## Get Started

---


### *Lighting in 15 mins*
---

This tutorial will walk you through the basics of using the torchssl-lightning framework. Let's get started by training a FixMatch model on CIFAR-10!

```python
import sys
sys.path.append('../')
from src import get_dataset, get_data_loader, net_builder, get_algorithm, get_config, Trainer
```
---

1. **Define Configs and Create Config**
```python

config = {
    'algorithm': 'fixmatch',
    'net': 'wrn_28_2',
    'use_pretrain': False,  # todo: add pretrain
    'pretrain_path': None,

    # optimization configs
    'epoch': 3,
    'num_train_iter': 150,
    'num_eval_iter': 50,
    'optim': 'SGD',
    'lr': 0.03,
    'momentum': 0.9,
    'batch_size': 64,
    'eval_batch_size': 64,

    # dataset configs
    'dataset': 'cifar10',
    'num_labels': 40,
    'num_classes': 10,
    'input_size': 32,
    'data_dir': './data',

    # algorithm specific configs
    'hard_label': True,
    'uratio': 3,
    'ulb_loss_ratio': 1.0,

    # device configs
    'gpu': 0,
    'world_size': 1,
    'distributed': False,
}
config = get_config(config)

```
---

2. **Create Model and Specify Algorithm**

```python
algorithm = get_algorithm(config,  net_builder(config.net, from_name=False), tb_log=None, logger=None)
```
---

3. **Create Dataset and Dataloader**

```python
# create dataset
dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels,
                           config.num_classes, data_dir=config.data_dir)
# create data loader for labeled training set
train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
# create data loader for unlabeled training set
train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
# create data loader for evaluation
eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size)
```
---

4. **Train**

```python
trainer = Trainer(config, algorithm)
trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
```
---

5. **Evaluate**

```python
trainer.evaluate(eval_loader)
```

---

6. **Predict**

```python
y_pred, y_logits = trainer.predict(eval_loader)

```
---
---
### Benchmarking Algorithms 

## Customized Usage

### customizing algorithm

### customizing dataset




