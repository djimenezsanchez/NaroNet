# NaroNet: objective-based learning of the tumor microenvironment from multiplex imaging.
Trained only with patient-level labels, NaroNet quantifies the phenotypes, neighborhoods, and neighborhood interactions that have the highest influence on the predictive task. This is the python implementation as described in our [*paper*](https://arxiv.org/abs/2103.05385).  

![alt text](https://github.com/djimenezsanchez/NaroNet/blob/main/images/folder_overview.gif)

### Installation
This package requires Python 3.6 (or newer)
Please first install TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer) by following the official instructions. For GPU support, it is crucial to install the specific versions of CUDA that are compatible with the respective version of TensorFlow and Pytorch.

To install NaroNet:
```sh
pip install NaroNet
```

### Usage
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment | Technique | Example Image | Description | 
| :-- | :-: | :-:| :-- |
| `POLE_example`| 7-plex fluorescence imaging | <img src="https://github.com/djimenezsanchez/NaroNet/blob/main/images/example_endometrial_crop.tif" title="example image fluo" width="120px" align="center">  | [Endometrial cancer tissue exampe (download Example_POLE.zip)](https://zenodo.org/record/4630664#.YFoGLa9KiUk). | 

