# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Code is adapted from https://github.com/MedMNIST/MedMNIST/

__version__ = "2.0.2"
import os
from os.path import expanduser
import warnings
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import trange
import torchvision.transforms as transforms
import copy
import random
from skimage.util import montage as skimage_montage

from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data
from .datasetbase import BasicDataset


def get_default_root():
    home = expanduser("~")
    dirpath = os.path.join(home, ".medmnist")

    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    except:
        warnings.warn("Failed to setup default root.")
        dirpath = None

    return dirpath


DEFAULT_ROOT = get_default_root()

HOMEPAGE = "https://github.com/MedMNIST/MedMNIST/"

INFO = {
    "pathmnist": {
        "python_class": "PathMNIST",
        "description":
        "The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 89996,
            "val": 10004,
            "test": 7180
        },
        "license": "CC BY 4.0"
    },
    "chestmnist": {
        "python_class": "ChestMNIST",
        "description":
        "The ChestMNIST is based on the NIH-ChestXray14 dataset, a dataset comprising 112,120 frontal-view X-Ray images of 30,805 unique patients with the text-mined 14 disease labels, which could be formulized as a multi-label binary-class classification task. We use the official data split, and resize the source images of 1×1024×1024 into 1×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/chestmnist.npz?download=1",
        "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        "task": "multi-label, binary-class",
        "label": {
            "0": "atelectasis",
            "1": "cardiomegaly",
            "2": "effusion",
            "3": "infiltration",
            "4": "mass",
            "5": "nodule",
            "6": "pneumonia",
            "7": "pneumothorax",
            "8": "consolidation",
            "9": "edema",
            "10": "emphysema",
            "11": "fibrosis",
            "12": "pleural",
            "13": "hernia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 78468,
            "val": 11219,
            "test": 22433
        },
        "license": "CC0 1.0"
    },
    "dermamnist": {
        "python_class": "DermaMNIST",
        "description":
        "The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3×600×450 are resized into 3×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/dermamnist.npz?download=1",
        "MD5": "0744692d530f8e62ec473284d019b0c7",
        "task": "multi-class",
        "label": {
            "0": "actinic keratoses and intraepithelial carcinoma",
            "1": "basal cell carcinoma",
            "2": "benign keratosis-like lesions",
            "3": "dermatofibroma",
            "4": "melanoma",
            "5": "melanocytic nevi",
            "6": "vascular lesions"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 7007,
            "val": 1003,
            "test": 2005
        },
        "license": "CC BY-NC 4.0"
    },
    "octmnist": {
        "python_class": "OCTMNIST",
        "description":
        "The OCTMNIST is based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−1,536)×(277−512). We center-crop the images and resize them into 1×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/octmnist.npz?download=1",
        "MD5": "c68d92d5b585d8d81f7112f81e2d0842",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization",
            "1": "diabetic macular edema",
            "2": "drusen",
            "3": "normal"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 97477,
            "val": 10832,
            "test": 1000
        },
        "license": "CC BY 4.0"
    },
    "pneumoniamnist": {
        "python_class": "PneumoniaMNIST",
        "description":
        "The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is binary-class classification of pneumonia against normal. We split the source training set with a ratio of 9:1 into training and validation set and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−2,916)×(127−2,713). We center-crop the images and resize them into 1×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/pneumoniamnist.npz?download=1",
        "MD5": "28209eda62fecd6e6a2d98b1501bb15f",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "pneumonia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 4708,
            "val": 524,
            "test": 624
        },
        "license": "CC BY 4.0"
    },
    "retinamnist": {
        "python_class": "RetinaMNIST",
        "description":
        "The RetinaMNIST is based on the DeepDRiD challenge, which provides a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training set with a ratio of 9:1 into training and validation set, and use the source validation set as the test set. The source images of 3×1,736×1,824 are center-cropped and resized into 3×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/retinamnist.npz?download=1",
        "MD5": "bd4c0672f1bba3e3a89f0e4e876791e4",
        "task": "ordinal-regression",
        "label": {
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 1080,
            "val": 120,
            "test": 400
        },
        "license": "CC BY 4.0"
    },
    "breastmnist": {
        "python_class": "BreastMNIST",
        "description":
        "The BreastMNIST is based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign, and malignant. As we use low-resolution images, we simplify the task into binary classification by combining normal and benign as positive and classifying them against malignant as negative. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images of 1×500×500 are resized into 1×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/breastmnist.npz?download=1",
        "MD5": "750601b1f35ba3300ea97c75c52ff8f6",
        "task": "binary-class",
        "label": {
            "0": "malignant",
            "1": "normal, benign"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 546,
            "val": 78,
            "test": 156
        },
        "license": "CC BY 4.0"
    },
    "bloodmnist": {
        "python_class": "BloodMNIST",
        "description":
        "The BloodMNIST is based on a dataset of individual normal cells, captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092 images and is organized into 8 classes. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images with resolution 3×360×363 pixels are center-cropped into 3×200×200, and then resized into 3×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/bloodmnist.npz?download=1",
        "MD5": "7053d0359d879ad8a5505303e11de1dc",
        "task": "multi-class",
        "label": {
            "0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "ig",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 11959,
            "val": 1712,
            "test": 3421
        },
        "license": "CC BY 4.0"
    },
    "tissuemnist": {
        "python_class": "TissueMNIST",
        "description":
        "We use the BBBC051, available from the Broad Bioimage Benchmark Collection. The dataset contains 236,386 human kidney cortex cells, segmented from 3 reference tissue specimens and organized into 8 categories. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. Each gray-scale image is 32×32×7 pixels, where 7 denotes 7 slices. We take maximum values across the slices and resize them into 28×28 gray-scale images.",
        "url":
        "https://zenodo.org/record/5208230/files/tissuemnist.npz?download=1",
        "MD5": "ebe78ee8b05294063de985d821c1c34b",
        "task": "multi-class",
        "label": {
            "0": "Collecting Duct, Connecting Tubule",
            "1": "Distal Convoluted Tubule",
            "2": "Glomerular endothelial cells",
            "3": "Interstitial endothelial cells",
            "4": "Leukocytes",
            "5": "Podocytes",
            "6": "Proximal Tubule Segments",
            "7": "Thick Ascending Limb"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 165466,
            "val": 23640,
            "test": 47280
        },
        "license": "CC BY 3.0"
    },
    "organamnist": {
        "python_class": "OrganAMNIST",
        "description":
        "The OrganAMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Axial (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in axial views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organamnist.npz?download=1",
        "MD5": "866b832ed4eeba67bfb9edee1d5544e6",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 34581,
            "val": 6491,
            "test": 17778
        },
        "license": "CC BY 4.0"
    },
    "organcmnist": {
        "python_class": "OrganCMNIST",
        "description":
        "The OrganCMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Coronal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in coronal views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organcmnist.npz?download=1",
        "MD5": "0afa5834fb105f7705a7d93372119a21",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13000,
            "val": 2392,
            "test": 8268
        },
        "license": "CC BY 4.0"
    },
    "organsmnist": {
        "python_class": "OrganSMNIST",
        "description":
        "The OrganSMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Sagittal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in sagittal views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organsmnist.npz?download=1",
        "MD5": "e5c39f1af030238290b9557d9503af9d",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13940,
            "val": 2452,
            "test": 8829
        },
        "license": "CC BY 4.0"
    },
    "organmnist3d": {
        "python_class": "OrganMNIST3D",
        "description":
        "The source of the OrganMNIST3D is the same as that of the Organ{A,C,S}MNIST. Instead of 2D images, we directly use the 3D bounding boxes and process the images into 28×28×28 to perform multi-class classification of 11 body organs. The same 115 and 16 CT scans as the Organ{A,C,S}MNIST from the source training set are used as training and validation set, respectively, and the same 70 CT scans as the Organ{A,C,S}MNIST from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organmnist3d.npz?download=1",
        "MD5": "21f0a239e7f502e6eca33c3fc453c0b6",
        "task": "multi-class",
        "label": {
            "0": "liver",
            "1": "kidney-right",
            "2": "kidney-left",
            "3": "femur-right",
            "4": "femur-left",
            "5": "bladder",
            "6": "heart",
            "7": "lung-right",
            "8": "lung-left",
            "9": "spleen",
            "10": "pancreas"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 972,
            "val": 161,
            "test": 610
        },
        "license": "CC BY 4.0"
    },
    "nodulemnist3d": {
        "python_class": "NoduleMNIST3D",
        "description":
        "The NoduleMNIST3D is based on the LIDC-IDRI, a large public lung nodule dataset, containing images from thoracic CT scans. The dataset is designed for both lung nodule segmentation and 5-level malignancy classification task. To perform binary classification, we categorize cases with malignancy level 1/2 into negative class and 4/5 into positive class, ignoring the cases with malignancy level 3. We split the source dataset with a ratio of 7:1:2 into training, validation and test set, and center-crop the spatially normalized images (with a spacing of 1mm×1mm×1mm) into 28×28×28.",
        "url":
        "https://zenodo.org/record/5208230/files/nodulemnist3d.npz?download=1",
        "MD5": "902d495e3d91ad1a7bcac1a6b58a8fa2",
        "task": "binary-class",
        "label": {
            "0": "benign",
            "1": "malignant"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1158,
            "val": 165,
            "test": 526
        },
        "license": "CC BY 3.0"
    },
    "adrenalmnist3d": {
        "python_class": "AdrenalMNIST3D",
        "description":
        "The AdrenalMNIST3D is a new 3D shape classification dataset, consisting of shape masks from 1,584 left and right adrenal glands (i.e., 792 patients). Collected from Zhongshan Hospital Affiliated to Fudan University, each 3D shape of adrenal gland is annotated by an expert endocrinologist using abdominal computed tomography (CT), together with a binary classification label of normal adrenal gland or adrenal mass. Considering patient privacy, we do not provide the source CT scans, but the real 3D shapes of adrenal glands and their classification labels. We calculate the center of adrenal and resize the center-cropped 64mm×64mm×64mm volume into 28×28×28. The dataset is randomly split into training/validation/test set of 1,188/98/298 on a patient level.",
        "url":
        "https://zenodo.org/record/5208230/files/adrenalmnist3d.npz?download=1",
        "MD5": "bbd3c5a5576322bc4cdfea780653b1ce",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "hyperplasia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1188,
            "val": 98,
            "test": 298
        },
        "license": "CC BY 4.0"
    },
    "fracturemnist3d": {
        "python_class": "FractureMNIST3D",
        "description":
        "The FractureMNIST3D is based on the RibFrac Dataset, containing around 5,000 rib fractures from 660 computed tomography 153 (CT) scans. The dataset organizes detected rib fractures into 4 clinical categories (i.e., buckle, nondisplaced, displaced, and segmental rib fractures). As we use low-resolution images, we disregard segmental rib fractures and classify 3 types of rib fractures (i.e., buckle, nondisplaced, and displaced). For each annotated fracture area, we calculate its center and resize the center-cropped 64mm×64mm×64mm image into 28×28×28. The official split of training, validation and test set is used.",
        "url":
        "https://zenodo.org/record/5208230/files/fracturemnist3d.npz?download=1",
        "MD5": "6aa7b0143a6b42da40027a9dda61302f",
        "task": "multi-class",
        "label": {
            "0": "buckle rib fracture",
            "1": "nondisplaced rib fracture",
            "2": "displaced rib fracture"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1027,
            "val": 103,
            "test": 240
        },
        "license": "CC BY-NC 4.0"
    },
    "vesselmnist3d": {
        "python_class": "VesselMNIST3D",
        "description":
        "The VesselMNIST3D is based on an open-access 3D intracranial aneurysm dataset, IntrA, containing 103 3D models (meshes) of entire brain vessels collected by reconstructing MRA images. 1,694 healthy vessel segments and 215 aneurysm segments are generated automatically from the complete models. We fix the non-watertight mesh with PyMeshFix and voxelize the watertight mesh with trimesh into 28×28×28 voxels. We split the source dataset with a ratio of 7:1:2 into training, validation and test set.",
        "url":
        "https://zenodo.org/record/5208230/files/vesselmnist3d.npz?download=1",
        "MD5": "2ba5b80617d705141f3f85627108fce8",
        "task": "binary-class",
        "label": {
            "0": "vessel",
            "1": "aneurysm"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1335,
            "val": 192,
            "test": 382
        },
        "license": "CC0 1.0"
    },
    "synapsemnist3d": {
        "python_class": "SynapseMNIST3D",
        "description":
        "The SynapseMNIST3D is a new 3D volume dataset to classify whether a synapse is excitatory or inhibitory. It uses a 3D image volume of an adult rat acquired by a multi-beam scanning electron microscope. The original data is of the size 100×100×100um^3 and the resolution 8×8×30nm^3, where a (30um)^3 sub-volume was used in the MitoEM dataset with dense 3D mitochondria instance segmentation labels. Three neuroscience experts segment a pyramidal neuron within the whole volume and proofread all the synapses on this neuron with excitatory/inhibitory labels. For each labeled synaptic location, we crop a 3D volume of 1024×1024×1024nm^3 and resize it into 28×28×28 voxels. Finally, the dataset is randomly split with a ratio of 7:1:2 into training, validation and test set.",
        "url":
        "https://zenodo.org/record/5208230/files/synapsemnist3d.npz?download=1",
        "MD5": "1235b78a3cd6280881dd7850a78eadb6",
        "task": "binary-class",
        "label": {
            "0": "inhibitory synapse",
            "1": "excitatory synapse"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1230,
            "val": 177,
            "test": 352
        },
        "license": "CC BY 4.0"
    }
}

SPLIT_DICT = {
    "train": "TRAIN",
    "val": "VALIDATION",
    "test": "TEST"
}  # compatible for Google AutoML Vision


def save2d(imgs, labels, img_folder,
           split, postfix, csv_path):
    return save_fn(imgs, labels, img_folder,
                   split, postfix, csv_path,
                   load_fn=lambda arr: Image.fromarray(arr),
                   save_fn=lambda img, path: img.save(path))


def montage2d(imgs, n_channels, sel):

    sel_img = imgs[sel]
    montage_arr = skimage_montage(sel_img, multichannel=(n_channels == 3))
    montage_img = Image.fromarray(montage_arr)

    return montage_img


def save3d(imgs, labels, img_folder,
           split, postfix, csv_path):
    return save_fn(imgs, labels, img_folder,
                   split, postfix, csv_path,
                   load_fn=load_frames,
                   save_fn=save_frames_as_gif)


def montage3d(imgs, n_channels, sel):

    montage_frames = []
    for frame_i in range(imgs.shape[1]):
        montage_frames.append(montage2d(imgs[:, frame_i], n_channels, sel))

    return montage_frames


def save_fn(imgs, labels, img_folder,
            split, postfix, csv_path,
            load_fn, save_fn):

    assert imgs.shape[0] == labels.shape[0]

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    if csv_path is not None:
        csv_file = open(csv_path, "a")

    for idx in trange(imgs.shape[0]):

        img = load_fn(imgs[idx])

        label = labels[idx]

        file_name = f"{split}{idx}_{'_'.join(map(str,label))}.{postfix}"

        save_fn(img, os.path.join(img_folder, file_name))

        if csv_path is not None:
            line = f"{SPLIT_DICT[split]},{file_name},{','.join(map(str,label))}\n"
            csv_file.write(line)

    if csv_path is not None:
        csv_file.close()


def load_frames(arr):
    frames = []
    for frame in arr:
        frames.append(Image.fromarray(frame))
    return frames


def save_frames_as_gif(frames, path, duration=200):
    assert path.endswith(".gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)


class MedMNIST(Dataset):

    flag = ...

    def __init__(self,
                 alg,
                 split,
                 transform=None,
                 transform_strong=None,
                 target_transform=None,
                 download=False,
                 as_rgb=False,
                 root=DEFAULT_ROOT,
                 is_ulb=False):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        '''
        self.alg = alg
        self.is_ulb = is_ulb
        self.strong_transform = transform_strong

        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found. ' +
                               ' You can set `download=True` to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == 'train':
            self.data = npz_file['train_images']
            self.targets = npz_file['train_labels'].reshape(-1)
        elif self.split == 'val':
            self.data = npz_file['val_images']
            self.targets = npz_file['val_labels'].reshape(-1)
        elif self.split == 'test':
            self.data = npz_file['test_images']
            self.targets = npz_file['test_labels'].reshape(-1)
        else:
            raise ValueError

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.ss'''
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"],
                         root=self.root,
                         filename="{}.npz".format(self.flag),
                         md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               HOMEPAGE)


class MedMNIST2D(MedMNIST, BasicDataset):

    def __sample__(self, idx):
        img, target = self.data[idx], self.targets[idx].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')
        return img, target

    def save(self, folder, postfix="png", write_csv=True):

        save2d(imgs=self.data,
               labels=self.targets,
               img_folder=os.path.join(folder, self.flag),
               split=self.split,
               postfix=postfix,
               csv_path=os.path.join(folder, f"{self.flag}.csv") if write_csv else None)

    def montage(self, length=20, replace=False, save_folder=None):

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(imgs=self.data,
                                n_channels=self.info['n_channels'],
                                sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(os.path.join(save_folder,
                                          f"{self.flag}_{self.split}_montage.jpg"))

        return montage_img


class MedMNIST3D(MedMNIST):

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.data[index], self.targets[index].astype(int)

        img = np.stack([img/255.]*(3 if self.as_rgb else 1), axis=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="gif", write_csv=True):

        assert postfix == "gif"

        save3d(imgs=self.data,
               labels=self.targets,
               img_folder=os.path.join(folder, self.flag),
               split=self.split,
               postfix=postfix,
               csv_path=os.path.join(folder, f"{self.flag}.csv") if write_csv else None)

    def montage(self, length=20, replace=False, save_folder=None):
        assert self.info['n_channels'] == 1

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_frames = montage3d(imgs=self.data,
                                   n_channels=self.info['n_channels'],
                                   sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_frames_as_gif(montage_frames,
                               os.path.join(save_folder,
                                            f"{self.flag}_{self.split}_montage.gif"))

        return montage_frames


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"


class OrganMNIST3D(MedMNIST3D):
    flag = "organmnist3d"


class NoduleMNIST3D(MedMNIST3D):
    flag = "nodulemnist3d"


class AdrenalMNIST3D(MedMNIST3D):
    flag = "adrenalmnist3d"


class FractureMNIST3D(MedMNIST3D):
    flag = "fracturemnist3d"


class VesselMNIST3D(MedMNIST3D):
    flag = "vesselmnist3d"


class SynapseMNIST3D(MedMNIST3D):
    flag = "synapsemnist3d"


# backward-compatible
OrganMNISTAxial = OrganAMNIST
OrganMNISTCoronal = OrganCMNIST
OrganMNISTSagittal = OrganSMNIST


def balanced_selection(total_data, total_targets, num_classes, per_class_data):
    select_index_set = np.zeros(num_classes * per_class_data, dtype=int) - 1
    label_counter = [0] * num_classes
    j = 0
    for i, label in enumerate(total_targets):
        if label_counter[label] != per_class_data:
            label_counter[label] += 1
            select_index_set[j] = i
            j += 1
        if label_counter == [per_class_data] * num_classes:
            break
    unselected_index_set = np.ones(total_targets.shape).astype(bool)
    unselected_index_set[select_index_set] = 0
    unselected_index_set, = np.where(unselected_index_set)

    selected_data = total_data[select_index_set]
    selected_targets = total_targets[select_index_set]
    unselected_data = total_data[unselected_index_set]
    unselected_targets = total_targets[unselected_index_set]
    return selected_data, selected_targets, unselected_data, unselected_targets


def get_medmnist(args, alg, dataset_name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    data_dir = os.path.join(data_dir, 'medmnist', dataset_name.lower())

    name2class = {
        "pathmnist": PathMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "bloodmnist": BloodMNIST,
        "tissuemnist": TissueMNIST,
        "organamnist": OrganAMNIST,
        "organcmnist": OrganCMNIST,
        "organsmnist": OrganSMNIST,
        "organmnist3d": OrganMNIST3D,
        "nodulemnist3d": NoduleMNIST3D,
        "adrenalmnist3d": AdrenalMNIST3D,
        "fracturemnist3d": FractureMNIST3D,
        "vesselmnist3d": VesselMNIST3D,
        "synapsemnist3d": SynapseMNIST3D,
    }

    dataset_mean = (0.5, 0.5, 0.5)
    dataset_std = (0.5, 0.5, 0.5)
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    n_labeled_per_class = int(num_labels // num_classes)

    transform_weak = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(img_size, padding=int(img_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(img_size, padding=int(img_size * (1 - crop_ratio)), padding_mode='reflect'),
        RandAugment(3, 5, exclude_color_aug=True),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    base_dataset = name2class[dataset_name](alg, split="train", root=data_dir, download=True, as_rgb=True)
    num_classes = len(INFO[dataset_name]["label"])

    train_targets = base_dataset.targets
    train_data = base_dataset.data  # np.ndarray, uint8
    assert len(train_targets) == len(train_data), "EuroSat dataset has an error!!!"

    # shuffle the dataset
    shuffle_index = list(range(len(train_targets)))
    np.random.shuffle(shuffle_index)
    total_targets = train_targets[shuffle_index]
    total_data = train_data[shuffle_index]

    train_labeled_data, train_labeled_targets, train_unlabeled_data, train_unlabeled_targets = split_ssl_data(args, total_data, total_targets, num_classes, 
                                                                                                              lb_num_labels=num_labels,
                                                                                                              ulb_num_labels=args.ulb_num_labels,
                                                                                                              lb_imbalance_ratio=args.lb_imb_ratio,
                                                                                                              ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                                                              include_lb_to_ulb=include_lb_to_ulb)
                                                                                                              
    if alg == 'fullysupervised':
        if len(train_unlabeled_data) == len(total_data):
            train_labeled_data = train_unlabeled_data 
            train_labeled_targets = train_unlabeled_targets
        else:
            train_labeled_data = np.concatenate([train_labeled_data, train_unlabeled_data], axis=0)
            train_labeled_targets = np.concatenate([train_labeled_targets, train_unlabeled_targets], axis=0)
    # construct datasets for training and testing
    train_labeled_dataset = name2class[dataset_name](alg, root=data_dir, split="train", transform=transform_weak, transform_strong=transform_strong, as_rgb=True)
    train_unlabeled_dataset = name2class[dataset_name](alg, root=data_dir, split="train", is_ulb=True, transform=transform_weak, transform_strong=transform_strong, as_rgb=True)
    train_labeled_dataset.data = train_labeled_data
    train_unlabeled_dataset.data = train_unlabeled_data
    train_labeled_dataset.targets = train_labeled_targets
    train_unlabeled_dataset.targets = train_unlabeled_targets
    val_dataset = []
    test_dataset = name2class[dataset_name](alg, root=data_dir, split="test", transform=transform_val, download=True, as_rgb=True)

    print(f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} "
          f"#Val: {len(val_dataset)} #Test: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
