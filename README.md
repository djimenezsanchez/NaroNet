# NaroNet: discovering tumor microenvironment elements from highly multiplexed images.
Trained only with subject-level labels, NaroNet uses deep learning to identify and annotate known as well as novel tumor microenvironment elements. This is the python implementation from our [*paper*](https://arxiv.org/abs/2103.05385).  

![alt text](https://github.com/djimenezsanchez/NaroNet/blob/main/images/folder_overview.gif)

### How it works?
A  

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



