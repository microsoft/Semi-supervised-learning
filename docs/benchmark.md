# Datasets

## CV Datasets

1. [CIFAR-100](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf): 100-class dataset with image size 32 x 32 pixels. For each class, there are 500 training samples and 100 testing samples.

1. [STL10](https://proceedings.mlr.press/v15/coates11a.html): 10-class dataset with image size of 96 x 96 pixels. Each class has 500 training samples and 800 testing samples. Besides labeled set, there is a unlabeled set with 100,000 samples.

1. [TissueMNIST](https://arxiv.org/abs/2010.14925): a medical dataset of human kidney cortex cells, segmented from 3 reference tissue specimens and organized into 8 categories. The total 236,386 training samples are split with a ratio of 7 : 1 : 2 into training (165,466 images), validation (23,640 images) and test set (47,280 images). Each gray-scale image is 28 x 28 pixels.

1. [SemiAves](https://arxiv.org/abs/2103.06937): Aves (birds) classification, where 3,959 images of 200 bird species are labeled and 26,640 images are unlabeled. The validation and test set contain 10 and 20 images respectively for each of the 200 categories in the labeled set.

1. [EuroSAT](https://arxiv.org/abs/1709.00029): covers Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27,000 labeled and geo-referenced samples. Training, validation and test set follows 6:2:2 ratio splits. 

1. [ImageNet](https://arxiv.org/abs/1409.0575): 1000-class, high resolution (224 x 224 pixels) recognition dataset. The number of images within each class ranges from 732 to 1300. The validation set consists of 50,000 images, which is evenly distributed across classes.

---

## NLP Datasets

1. [IMDB](https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf): is a binary sentiment classification dataset. There are 25,000 reviews for training and 25,000 for test. IMDB is class balanced which means the positive and negative reviews have the same number both for training and test. For USB, we draw 12,500 samples and 1,000 samples per class from training samples to form the training dataset and validation dataset respectively. The test dataset is unchanged.

1. [Amazon Review](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf): is a sentiment classification dataset. There are 5 classes (scores). Each class (score) contains 600,000 training samples and 130,000 test samples. For USB, we draw 50,000 samples and 5,000 samples per class from training samples to form the training dataset and validation dataset respectively. The test dataset is unchanged.

1. [Yelp Review](http://www.yelp.com/dataset\_challenge): is a sentiment classification dataset with 5 classes (scores). Each class (score) contains 130,000 training samples and 10,000 test samples. For USB, we draw 50,000 samples and 5,000 samples per class from training samples to form the training dataset and validation dataset respectively. The test dataset is unchanged.

1. [AG News](https://arxiv.org/abs/1509.01626): is a news topic classification dataset containing 4 classes. Each class contains 30,000 training samples and 1,900 test samples. For USB, we draw 25,000 samples and 2,500 samples per class from training samples to form the training dataset and validation dataset respectively. The test dataset is unchanged.

1. [Yahoo! Answer](https://dl.acm.org/doi/10.5555/1620163.1620201): is a topic classification dataset has 10 categories. Each class contains 140,000 training samples and 6,000 test samples. For USB, we draw 50,000 samples and 5,000 samples per class from training samples to form the training dataset and validation dataset respectively. The test dataset is unchanged.
---
## Audio Datasets

1. *GTZAN*: is collected for music genre classification of 10 classes and 100 audio recordings for each class. The maximum length of the recordings is 30 seconds and the original sampling rate is 22,100 Hz. We split 7,000 samples for training, 1,500 for validation, and 1,500 for testing. All recordings are re-sampled at 16,000 Hz.

1. [UrbanSound8k](https://dl.acm.org/doi/10.1145/2647868.2655045): contains 8,732 labeled sound events of urban sounds of 10 classes, with the maximum length of 4 seconds. The original sampling rate of the audio recordings is 44,100 and we re-sample it to 16,000. It is originally divided into 10 folds, where we use the first 8 folds of 7,079 samples as training set, and the last two folds as validation set of size 816 and testing set of size 837 respectively.

1. [FSDNoisy18k](https://arxiv.org/abs/1901.01189): is a sound event classification dataset across 20 classes. It consists of a small amount of manually labeled data - 1,772 and a large amount of noisy data - 15,813 which is treated as unlabeled data in our paper. The original sample rate is 44,100 Hz, and the length of the recordings lies between 3 seconds and 30 seconds. We use the testing set provided for evaluation, which contains 947 samples.

1. *Keyword Spotting*: is one of the tasks in Superb[30] for classifying the keywords. It contains speech utterances of a maximum length of 1 second and the sampling rate of 16,000. The training, validation, and testing set contain 18,538; 2,577; 2,567 recordings, respectively. For pre-processing, we remove the silence and unknown labels from the dataset. 

1. [ESC-50](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT): is a dataset containing 2,000 environmental audio recordings for 50 sound classes. The maximum length of the recordings is 5 seconds and the original sampling rate is 44,100. We split 1,200 samples as training data, 400 as validation data, and 400 as testing data. We also re-sample the audio recordings to 16,000 Hz during pre-processing.
