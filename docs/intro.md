# USB
>USB is a unified semi-supervised learning benchmark. With USB, we open-source a modular and extensible codebase including labeled data, unlabeled data, and 13 SSL algorithms for standardization of data sampling, implementation, evaluation, and ablation of SSL methods. To enable consistent evaluation over multiple datasets from multiple domains.

---

## Datasets
6 CV datasets, 6 NLP datasets, and 6 Speech datasets are selected to jointly form the comprehensive and challenging USB.

---

### CV Datasets

1. CIFAR-10 /CIFAR-100
2. SVHN
3. STL10
4. ImageNet
5. MedMNIST
6. aves?

---

### NLP Datasets

---
### Speech Datasets

---
---

## Algorithms

In addtion to fully-supervised method (as a baseline), USB supports the following popular algorithms:
1. UDA (NeurIPS 2020)[6]
1. VAT (TPAMI 2018)[4]
1. SoftMatch
1. SimMatch
1. ReMixMatch (ICLR 2019)[7]
1. Pseudo-Label (ICML 2013)[3]
1. PiModel (NeurIPS 2015)[1]
1. MPL
1. MixMatch (NeurIPS 2019) [5]
1. MeanTeacher (NeurIPS 2017)[2]
1. FlexMatch (NeurIPS 2020)[9]
1. FixMatch (NeurIPS 2021)[8]
1. Dash
1. CrMatch
1. CoMatch
1. AdaMatch



### References

[1] Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, and Tapani Raiko.  Semi-supervised learning with ladder networks. InNeurIPS, pages 3546–3554, 2015.

[2] Antti Tarvainen and Harri Valpola.  Mean teachers are better role models:  Weight-averagedconsistency targets improve semi-supervised deep learning results. InNeurIPS, pages 1195–1204, 2017.

[3] Dong-Hyun Lee et al. Pseudo-label: The simple and efficient semi-supervised learning methodfor  deep  neural  networks.   InWorkshop  on  challenges  in  representation  learning,  ICML,volume 3, 2013.

[4] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training:a regularization method for supervised and semi-supervised learning.IEEE TPAMI, 41(8):1979–1993, 2018.

[5] David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and ColinRaffel. Mixmatch: A holistic approach to semi-supervised learning.NeurIPS, page 5050–5060,2019.

[6] Qizhe Xie, Zihang Dai, Eduard Hovy, Thang Luong, and Quoc Le. Unsupervised data augmen-tation for consistency training.NeurIPS, 33, 2020.

[7] David Berthelot, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang,and Colin Raffel.   Remixmatch:  Semi-supervised learning with distribution matching andaugmentation anchoring. InICLR, 2019.

[8] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raf-fel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li.  Fixmatch:  Simplifying semi-supervised learning with consistency and confidence.NeurIPS, 33, 2020.

[9] Bowen Zhang, Yidong Wang, Wenxin Hou, Hao wu, Jindong Wang, Okumura Manabu, and Shinozaki Takahiro. FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. NeurIPS, 2021.
```

