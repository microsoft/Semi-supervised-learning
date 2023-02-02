# Benchmark Results

This folder contains benchmark results for 14 SSL algorithms. We report the best error rate（top-1-1 and top-1-20 results can be found in drive）. Each setting runs 3 different random seeds and computes the average performance with standard deviation.

## Reproduction on Classic CV Benchmark

### Datasets

| Dataset   | #Label       | #Training data | #Test data | #Class |
| --------- | ------------ | -------------- | ---------- | ------ |
| CIFAR-10  | 4 / 25 / 400 | 50,000         | 10,000     | 10     |
| CIFAR-100 | 4 / 25 / 100 | 50,000         | 10,000     | 100    |
| SVHN      | 4 / 25 / 100 | 604,388        | 26,032     | 10     |
| STL-10    | 4 / 25 / 100 | 100,000        | 10,000     | 10     |

### Results

Please see [Classic CV](classic_cv.xls) for the complete table of results.

## USB CV Results

### Datasets

| Dataset     | #Label  | #Training data  | #Test data | #Class |
| ----------- | ------- | --------------- | ---------- | ------ |
| CIFAR-100   | 2 / 4   | 50,000          | 10,000     | 100    |
| STL-10      | 4 / 10  | 5,000 / 100,000 | 8,000      | 10     |
| EuroSat     | 2 / 4   | 16,200          | 5,400      | 10     |
| TissueMNIST | 10 / 50 | 165,466         | 47,280     | 8      |
| Semi-Aves   | 15-53   | 3,959 / 26,640  | 4,000      | 200    |

### Results

Please see [USB CV](usb_cv.xls) for the complete table of results.

## USB NLP Results

### Datasets

| Dataset       | #Label   | #Training data | #Val data | #Test data | #Class |
| ------------- | -------- | -------------- | --------- | ---------- | ------ |
| IMDB          | 10 / 50  | 23,000         | 2,000     | 25,000     | 2      |
| Amazon Review | 50 / 200 | 250,000        | 25,000    | 65,000     | 5      |
| Yelp Review   | 50 / 200 | 250,000        | 25,000    | 50,000     | 5      |
| AG News       | 10 / 50  | 100,000        | 10,000    | 7,600      | 4      |
| Yahoo! Answer | 50 / 200 | 500,000        | 50,000    | 60,000     | 10     |

### Results

Please see [USB NLP](usb_nlp.xls) for the complete table of results.

## USB Audio Results

### Datasets

| Dataset          | #Label  | #Training data | #Val data | #Test data | #Class |
| ---------------- | ------- | -------------- | --------- | ---------- | ------ |
| Keyword Spotting | 5 / 20  | 18,538         | 2,577     | 2,567      | 10     |
| ESC-50           | 5 / 10  | 1,200          | 400       | 400        | 50     |
| UrbanSound8k     | 10 / 40 | 7,079          | 816       | 837        | 10     |
| FSDnoisy18k      | 52-171  | 1,772 / 15,813 | -         | 947        | 20     |
| GTZAN            | 10 / 40 | 7,000          | 1,500     | 1,500      | 10     |

### Results

Please see [USB Audio](usb_audio.xls) for the complete table of results.
