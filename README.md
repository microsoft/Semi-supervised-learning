<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://github.com/microsoft/Semi-supervised-learning">
    <img src="figures/logo.png" alt="Logo" width="400" height="160">
  </a>

<h3 align="center">USB</h3>

<p align="center">
    A Unified Semi-supervised Learning Benchmark for CV, NLP, Audio
    <!-- <br />
    <a href="https://github.com/microsoft/Semi-supervised-learning"><strong>Explore the docs »</strong></a>
    <br /> -->
    <br />
    <a href="https://github.com/microsoft/Semi-supervised-learning">Paper</a>
    ·
    <a href="https://github.com/microsoft/Semi-supervised-learning">Benchmark</a>
    ·
    <a href="https://github.com/microsoft/Semi-supervised-learning">Demo</a>
    ·
    <a href="https://github.com/microsoft/Semi-supervised-learning">Docs</a>
    ·
    <a href="https://github.com/microsoft/Semi-supervised-learning/issues">Issue</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#news-and-updates">News and Updates</a></li>
    <li><a href="#intro">Introduction</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#benchmark-results">Benchmark Results</a></li>
    <li><a href="#model-zoo">Model Zoo</a></li>
    <li><a href="#contributing">Community</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- News and Updates -->

## News and Updates

- [08/21/2022] USB has been released!

<!-- Introduction -->

## Introduction

USB is a Pytorch-based Python package for Semi-Supervised Learning (SSL). It is easy-to-use/extend, affordable, and comprehensive for developing and evaluating SSL algorithms. USB provides the implementation of 14 SSL algorithms based on Consistency Regularization, and 15 tasks for evaluation from CV, NLP, and Audio domain.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

This is an example of how to set up USB locally.
To get a local copy up, running follow these simple example steps.

### Prerequisites

USB is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:

```sh
conda create --name usb python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Installation

We provide a Python package of USB for users who want to start training/testing the supported SSL algorithms on their data quickly:

```sh
pip install usb
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Development

You can also develop your own SSL algorithm and evaluate it by cloning USB:

```sh
git clone https://github.com/microsoft/Semi-supervised-learning.git
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

USB is easy to use and extend. Going through the belowing examples will help you faimiliar with USB for quick use, evaluate an exsiting SSL algorithm on your own dataset, or developing new SSL algorithms.

### Quick Start with USB package

<!-- TODO: add quick start example and refer lighting notebook -->

Please see [Installation](#installation) to install USB first. We provide colab tutorials for:

- [Beginning example](notebooks/Beginner_Example.ipynb)
- [Customize datasets](notebooks/Custom_Dataset.ipynb)

### Training

Here is an example to train FixMatch on CIFAR-100 with 200 labels. Trianing other supported algorithms (on other datasets with different label settings) can be specified by a config file:

```sh
python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0.yaml
```

### Evaluation

After trianing, you can check the evaluation performance on training logs, or running evaluation script:

```
python eval.py --dataset cifar100 --num_classes 100 --load_path /PATH/TO/CHECKPOINT
```

### Develop

Check the developing documentation for creating your own SSL algorithm!

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- BENCHMARK RESULTS -->

## Benchmark Results

Please refer to [Results](./results) for benchmark results on different tasks.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MODEL ZOO -->

## Model Zoo

TODO: add pre-trained models.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->

## TODO

- [ ] Add docker
- [ ] Finish Readme
- [ ] Compile docs and add usage example in docs
- [ ] Check Notebooks Create Colab Notebooks
- [ ] Updating SUPPORT.MD with content about this project's support experience
- [ ] Multi-language Support
  - [ ] Chinese

See the [open issues](https://github.com/microsoft/Semi-supervised-learning/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

If you have a suggestion that would make USB better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the project
2. Create your branch (`git checkout -b your_name/your_branch`)
3. Commit your changes (`git commit -m 'Add some features'`)
4. Push to the branch (`git push origin your_name/your_branch`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- TRADEMARKS -->

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft&#39;s Trademark &amp; Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Community and Contact

The USB comunity is maintained by:

- Yidong Wang (),
- Hao Chen (haoc3@andrew.cmu.edu), Carnegie Mellon University
- Yue Fan (),
- Wenxin Hou (),
- Ran Tao (),
- Jindong Wang (),

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITE -->

## Citing USB

Please cite us if you fine USB helpful for your project/paper:

```
@misc{usb2022,
  doi = {10.48550/ARXIV.2208.07204},
  url = {https://arxiv.org/abs/2208.07204},
  author = {Wang, Yidong and Chen, Hao and Fan, Yue and Sun, Wang and Tao, Ran and Hou, Wenxin and Wang, Renjie and Yang, Linyi and Zhou, Zhi and Guo, Lan-Zhe and Qi, Heli and Wu, Zhen and Li, Yu-Feng and Nakamura, Satoshi and Ye, Wei and Savvides, Marios and Raj, Bhiksha and Shinozaki, Takahiro and Schiele, Bernt and Wang, Jindong and Xie, Xing and Zhang, Yue},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {USB: A Unified Semi-supervised Learning Benchmark},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

We thanks the following projects for reference of creating USB:

- [TorchSSL](https://github.com/TorchSSL/TorchSSL)
- [FixMatch](https://github.com/google-research/fixmatch)
- [CoMatch](https://github.com/salesforce/CoMatch)
- [SimMatch](https://github.com/KyleZheng1997/simmatch)
- [HuggingFace](https://huggingface.co/docs/transformers/index)
- [Pytorch Lighting](https://github.com/Lightning-AI/lightning)
- [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[contributors-url]: https://github.com/microsoft/Semi-supervised-learning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[forks-url]: https://github.com/microsoft/Semi-supervised-learning/network/members
[stars-shield]: https://img.shields.io/github/stars/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[stars-url]: https://github.com/microsoft/Semi-supervised-learning/stargazers
[issues-shield]: https://img.shields.io/github/issues/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[issues-url]: https://github.com/microsoft/Semi-supervised-learning/issues
[license-shield]: https://img.shields.io/github/license/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[license-url]: https://github.com/microsoft/Semi-supervised-learning/blob/main/LICENSE.txt
