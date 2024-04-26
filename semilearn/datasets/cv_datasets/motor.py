import os
# import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

def get_motor(args, alg, name, num_labels, num_classes, data_dir = './data', include_lb_to_ulb=True):
    data_dir = os.path.join(data_dir, name.lower())

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 7),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 7),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])
    
    lb_data, lb_targets = [], []
    for label in os.listdir(os.path.join(data_dir, 'train')):
        path = os.path.join(data_dir, 'train', label)
        imagenames = os.listdir(path)
        imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
        targets = [int(label)]*len(os.listdir(path))    
        lb_data.extend(imagepaths)
        lb_targets.extend(targets)
    test_data, test_targets = [], []
    for label in os.listdir(os.path.join(data_dir, 'test')):
        path = os.path.join(data_dir, 'test', label)
        imagenames = os.listdir(path)
        imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
        targets = [int(label)]*len(os.listdir(path))    
        test_data.extend(imagepaths)
        test_targets.extend(targets)
    ulb_data, ulb_targets = [], []
    if alg != 'fullysupervised':
        path = os.path.join(data_dir, 'unlabel')
        imagenames = os.listdir(path)
        imagepaths = [os.path.join(path, imagename) for imagename in imagenames]  
        ulb_data = imagepaths
              
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
              
    lb_dset = MotorDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, None, False)
    ulb_dset = MotorDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    eval_dset = MotorDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)
    return lb_dset, ulb_dset, eval_dset

class MotorDataset(BasicDataset):
    def __sample__(self, idx):
        path =  self.data[idx]
        img = Image.open(path).convert('RGB')
        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target