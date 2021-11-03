# NaroNet: discovery of tumor microenvironment elements from highly multiplexed images.
***TL;DR:*** NaroNet is an interpretable method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to predict subject-level labels. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding patient-level labels. See our [*paper*](https://arxiv.org/abs/2103.05385).  

![alt text](https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview.gif)

##  
[Installation](#Installation) • [Usage](#Usage) • [Patch Contrastive Learning](#Patch Contrastive Learning) • [NaroNet](#NaroNet) • [BioInsights](#BioInsights) • [Cite](#reference)

### Installation
This package requires Python 3.6 (or newer)
Please first install TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer) by following the official instructions. For GPU support, it is crucial to install the specific versions of CUDA that are compatible with the respective version of TensorFlow and Pytorch.

To install NaroNet:
```sh
pip install NaroNet
```

### Usage
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment name | Example Image | Dataset link | Run in google colab |
| :-- | :-:| :-- | :-- |
| Discover tumoral differences between patient types (POLE gene mutated vs. POLE gene non-mutated) | <img src="https://github.com/djimenezsanchez/NaroNet/blob/main/images/example_endometrial_crop.png" title="example image fluo" width="320px" align="center">  | [Endometrial cancer tissue example (download Example_POLE.zip)](https://zenodo.org/record/4630664#.YFoGLa9KiUk). |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimenezsanchez/NaroNet/blob/main/examples/google_colab_example.ipynb?authuser=1) |

### Patch Contrastive Learning

### NaroNet

### BioInsights

### Citation



