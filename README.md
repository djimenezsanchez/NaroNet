# NaroNet: discovery of tumor microenvironment elements from highly multiplexed images.
***TL;DR:*** NaroNet is an end-to-end interpretable learning method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to predict subject-level labels. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding patient-level labels. See our [*paper*](https://arxiv.org/abs/2103.05385).  

![alt text](https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview.gif)

##  
[Requirements and installation](#Requirements-and-installation) • [Patch Contrastive Learning](#Patch-Contrastive-Learning) • [NaroNet](#NaroNet) • [BioInsights](#BioInsights) • [Cite](#reference) • [Demo](#Demo) 

### Requirements and installation
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 4 on GPU server, and Nvidia P100, K80 GPUs on Google Cloud)

We recommend installing [*anaconda*](https://www.anaconda.com/distribution/) and creating a new environment with TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer). For GPU support, it is crucial to install the specific versions of CUDA that are compatible with the respective version of TensorFlow and Pytorch.

Once inside the created environment, install pytorch-geometric where ${CUDA} and ${TORCH} should be replaced by the specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111, cu113) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0):
```sh
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

To install NaroNet:
```sh
pip install NaroNet
```

### Patch Contrastive Learning

### NaroNet

### BioInsights

### Demo
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment name | Example Image | Dataset link | Run in google colab |
| :-- | :-:| :-- | :-- |
| Discover tumoral differences between patient types (POLE gene mutated vs. POLE gene non-mutated) | <img src="https://github.com/djimenezsanchez/NaroNet/blob/main/images/example_endometrial_crop.png" title="example image fluo" width="320px" align="center">  | [Endometrial cancer tissue example (download Example_POLE.zip)](https://zenodo.org/record/4630664#.YFoGLa9KiUk). |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimenezsanchez/NaroNet/blob/main/examples/google_colab_example.ipynb?authuser=1) |


### Citation



