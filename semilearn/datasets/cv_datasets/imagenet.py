# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# fixmatch, flexmatch, pseudolabel, vat

import os
import gc
import copy
import json
import random
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    data_dir = os.path.join(data_dir, name.lower())

    train_imglist = '/home/ubuntu/V-SAFE/supervised/data/train_imagenet200.txt'
    val_imglist = '/home/ubuntu/V-SAFE/supervised/data/val_imagenet200.txt'
    imgpath = '/ephemeral/'
    dataset = ImagenetDataset(root=imgpath, transform=transform_weak, ulb=False, alg=alg, imglist_pth=train_imglist)
    label_perclass = num_labels // (max(dataset.targets)+1)

    lb_dset = ImagenetDataset(root=imgpath, transform=transform_weak, ulb=False, alg=alg, imglist_pth=train_imglist, label_perclass=label_perclass)
    breakpoint()

    ulb_dset = ImagenetDataset(root=imgpath, transform=transform_weak, alg=alg, imglist_pth=train_imglist, ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)

    eval_dset = ImagenetDataset(root=imgpath, transform=transform_val, alg=alg, imglist_pth=val_imglist, ulb=False)

    # if args.use_noise:
    #     noise_path = args.noise_path
    #     ulb_dset = ImagenetDataset(root="", transform=transform_weak, alg=alg, imglist_pth=noise_path, ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    ood_count = 0
    for lb in lb_dset.targets:
        lb_count[lb] += 1
    for ulb in ulb_dset.targets:
        if ulb >= 0:
            ulb_count[ulb] += 1
        if ulb == -1:
            ood_count += 1
    save_dir = os.path.join(args.save_dir, args.save_name)
    noise_name = "None"
    with open(os.path.join(save_dir, f'{noise_name}.txt'), 'w') as f:
        f.write("Dataset: {}\n".format(noise_name))
        f.write("lb_count: {}\n".format(lb_count))
        f.write("ulb_count: {}\n".format(ulb_count + [ood_count]))
        f.write("OOD unlabeled images: {}\n".format(ood_count))
        f.close()

    return lb_dset, ulb_dset, eval_dset
    


class ImagenetDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, imglist_pth=None, medium_transform=None, strong_transform=None, label_perclass=-1, include_lb_to_ulb=True, lb_index=None):
        self.alg = alg
        self.is_ulb = ulb
        self.label_perclass = label_perclass
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index

        if imglist_pth is not None:
            samples = self._make_dataset_from_list(imglist_pth)
        else:
            raise ValueError("You must provide imglist_pth for ImagenetDataset")

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 samples in {imglist_pth}")
        
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.loader = default_loader

        # classes, class_to_idx = self.find_classes(self.root)
        # self.classes = classes
        # self.class_to_idx = class_to_idx

        unique_targets = sorted(set(self.targets))
        self.classes = [str(c) for c in unique_targets]
        self.class_to_idx = {str(c): c for c in unique_targets}


        self.medium_transform = medium_transform
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"


    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    
class ImagenetDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, imglist_pth=None, lb_list_txt=None, medium_transform=None, strong_transform=None, label_perclass=-1, include_lb_to_ulb=True, lb_index=None):
        self.alg = alg
        self.is_ulb = ulb
        self.label_perclass = label_perclass
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index
        
        # 新增：接收指定 index 的 txt 檔案路徑
        self.lb_list_txt = lb_list_txt 

        if imglist_pth is not None:
            samples = self._make_dataset_from_list(imglist_pth)
        else:
            raise ValueError("You must provide imglist_pth for ImagenetDataset")

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 samples in {imglist_pth}")
        
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.loader = default_loader

        unique_targets = sorted(set(self.targets))
        self.classes = [str(c) for c in unique_targets]
        self.class_to_idx = {str(c): c for c in unique_targets}

        self.medium_transform = medium_transform
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"

    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target
    
    def _make_dataset_from_list(self, imglist_pth):
        """
        imglist_pth: 原始的大全 txt (e.g., train_imagenet200.txt)
        格式: 'imagenet_1k/train/n04372370/n04372370_9138.JPEG 844'
        """
        instances = []
        buckets = {} 

        with open(imglist_pth, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # === 步驟 1: 建立 Buckets 並保留原始 Index ===
        for i, line in enumerate(lines):
            path, target = line.split()
            target = int(target)
            full_path = os.path.join(self.root, path)
            
            if os.path.isfile(full_path):
                # 這裡存入 tuple: (完整路徑, label, 原始index, 相對路徑)
                # 相對路徑用來做 string matching (如果有的話)
                buckets.setdefault(target, []).append((full_path, target, i, path))

        lb_idx = {}
        saved_idxs = [] # 用來收集這次選到的 index

        # === 分支 A: 如果有指定要讀取的 labeled txt 檔案 (Loading Mode) ===
        # 假設 self.lb_list_txt 是在 __init__ 傳進來的路徑
        if hasattr(self, 'lb_list_txt') and self.lb_list_txt is not None and not self.is_ulb:
            print(f"Loading specific labeled data from: {self.lb_list_txt}")
            with open(self.lb_list_txt, 'r') as f:
                # 假設讀進來的是一行一個相對路徑，或是 index
                # 這裡示範讀取相對路徑 (比較穩健)
                target_paths = set([ln.strip().split()[0] for ln in f if ln.strip()])
            
            for cls, items in buckets.items():
                for full_path, target, original_idx, rel_path in items:
                    # 比對相對路徑
                    if rel_path in target_paths:
                        instances.append((full_path, target))
                        lb_idx.setdefault(cls, []).append(os.path.basename(full_path))
            
            if len(instances) == 0:
                 print("Warning: No labeled data matches found in the provided list!")

        # === 分支 B: 隨機取樣並存檔 (Sampling & Saving Mode) ===
        elif self.label_perclass > 0 and not self.is_ulb:
            print(f"Randomly sampling {self.label_perclass} per class...")
            
            for cls, items in buckets.items():
                # items 是 (full_path, target, original_idx, rel_path)
                k = min(self.label_perclass, len(items))
                chosen = random.sample(items, k)
                
                # 1. 加入 instances 給 dataset 使用 (只需路徑和 label)
                instances.extend([(x[0], x[1]) for x in chosen])
                
                # 2. 紀錄檔名 (原本 semilearn 的邏輯)
                lb_idx[cls] = [os.path.basename(x[0]) for x in chosen]
                
                # 3. 收集原始 index (x[2] 是我們上面存的 i)
                saved_idxs.extend([x[2] for x in chosen])

            # --- 存檔邏輯 ---
            save_name = 'lb_labels10000_1_seed0_idx.txt'
            print(f"Saving sampled indices to {save_name} ...")
            
            # 排序讓檔案好看一點 (Optional)
            saved_idxs.sort()
            
            with open(save_name, 'w') as f:
                for idx in saved_idxs:
                    f.write(f"{idx}\n")
            # ----------------

        # === 分支 C: 全部使用 (Unlabeled data 或 Validation set) ===
        else:
            for cls, items in buckets.items():
                # items 是 (full_path, target, original_idx, rel_path)
                instances.extend([(x[0], x[1]) for x in items])
            lb_idx = {}

        gc.collect()
        self.lb_idx = lb_idx
        return instances
