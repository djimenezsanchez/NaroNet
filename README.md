# NaroNet: objective-based learning of the tumor microenvironment from multiplex imaging.
Trained only with patient-level labels, NaroNet quantifies the phenotypes, neighborhoods, and neighborhood interactions that have the highest influence on the predictive task. This is the python implementation as described in our [*paper*](https://arxiv.org/abs/2103.05385).  

![alt text](https://github.com/djimenezsanchez/NaroNet/blob/main/models/MethodDescription.png?raw=true)

### Installation
This package requires Python 3.6 (or newer)
Please first install TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer) by following the official instructions. For GPU support, it is crucial to install the specific versions of CUDA that are compatible with the respective version of TensorFlow and Pytorch.

To install NaroNet:
```sh
pip install NaroNet
```

### Usage
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment | Markers | Image format | Example Image | Description | 
| :-- | :-: | :-:| :-:| :-- |
| `POLE_example`| Fluorescence | 2D single channel| <img src="https://github.com/mpicbg-csbd/stardist/raw/master/images/example_fluo.jpg" title="example image fluo" width="120px" align="center">       | *Versatile (fluorescent nuclei)* and [subset of the DSB 2018 nuclei segmentation challenge dataset](https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip). | 
