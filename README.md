# NaroNet: discovery of tumor microenvironment elements from highly multiplexed images.
***TL;DR:*** NaroNet is an end-to-end interpretable learning method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to predict subject-level labels. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding patient-level labels. See our [*paper*](https://arxiv.org/abs/2103.05385).  

![alt text](https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview.gif)

##  
[Requirements and installation](#Requirements-and-installation) • [Preparing datasets](#Preparing-datasets) • [Patch Contrastive Learning](#Patch-Contrastive-Learning) • [NaroNet](#NaroNet) • [BioInsights](#BioInsights) • [Cite](#reference) • [Demo](#Demo) 

### Requirements and installation
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 4 on GPU server, and Nvidia P100, K80 GPUs on Google Cloud)

To install NaroNet we recommend creating a new [*anaconda*](https://www.anaconda.com/distribution/) environment with TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer). For GPU support, it is crucial to install the specific versions of CUDA that are compatible with the respective version of TensorFlow and Pytorch.

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

### Preparing datasets
When NaroNet is executed it expects the target folder (e.g., 'DATASET_DATA_DIR') to be organized as follows:

```bash
DATASET_DATA_DIR/
    └──Raw_Data/
        ├── Images/
                ├── image_1.tiff
                ├── image_2.tiff
                └── ...
        └── Experiment_Information/
                ├── Channels.txt                
                ├── Image_Labels.xlsx
		└── Patient_to_Image.xlsx (Optional)
		
```
Namely, each dataset is expected to be a subfolder (e.g., 'Raw_Data') under DATA_ROOT_DIR. In the 'Raw_Data/Images' our method assumes multiplex image data (i.e., multi-tiff). 
In the 'Raw_Data/Experiment_Information' it is expected to have two files:
* Channels.txt contains in each row the name of each marker/channel present in the multiplex image. In case the name of the row is 'None' it will be ignored and not loaded from the raw image.
```bash
Marker_1
Marker_2 
None
Marker_4    
```

* Image_Labels.xlsx contains the image-level labels. In column 'Image_Names' each row specifies one image. The next columns (e.g., 'Type_1', 'Type_2', etc.) specify image-level information, where 'None' means that there is no information available for this subject and therefore it has to be excluded from the experiment. 

| Image_Names | Type_1 | Type_2 | 
| :-- | :-:| :-: |
| Image_1.tiff | A  | X |
| Image_2.tiff | None  | Y |
| ... | ... | ... |

* Patient_to_Image.xlsx contains a table that specifies the image-level labels. In column 'Image_Names' each row specifies one image. The next columns (e.g., 'Type_1', 'Type_2', etc.) specify image-level information, where 'None' means that 

Datasets are also expected to be prepared in a csv format containing at least 3 columns: **case_id**, **slide_id**, and 1 or more labels columns for the slide-level labels. Each **case_id** is a unique identifier for a patient, while the **slide_id** is a unique identifier for a slide that correspond to the name of an extracted feature .pt file. This is necessary because often one patient has multiple slides, which might also have different labels. When train/val/test splits are created, we also make sure that slides from the same patient do not go to different splits. The slide ids should be consistent with what was used during the feature extraction step. We provide 2 dummy examples of such dataset csv files in the **dataset_csv** folder: one for binary tumor vs. normal classification (task 1) and one for multi-class tumor_subtyping (task 2). 

Dataset objects used for actual training/validation/testing can be constructed using the **Generic_MIL_Dataset** Class (defined in **datasets/dataset_generic.py**). Examples of such dataset objects passed to the models can be found in both **main.py** and **eval.py**. 

For training, look under main.py:
```python 
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_feat_resnet'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            label_col = 'label',
                            ignore=[])
```
The user would need to pass:
* csv_path: the path to the dataset csv file
* data_dir: the path to saved .pt features
* label_dict: a dictionary that maps labels in the label column to numerical values
* label_col: name of the label column (optional, by default it's 'label')
* ignore: labels to ignore (optional, by default it's an empty list)

Finally, the user should add this specific 'task' specified by this dataset object in the --task arguments as shown below:

```python
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
```


### Patch Contrastive Learning (PCL)
The goal of the first step of our pipeline is to convert each high-dimensional  multiplex  image  of  the  cohort  into a list of low-dimensional embedding vectors. To this end, each image is divided into patches -our basic units of representation containin one or two cells of the tissue-, and each patch is converted by the PCL module -a properly trained CNN- into a low-dimensional vector that embeds both the morphological and spectral information of the patch.

Our method assumes that multiplex image data (i.e., tiff) are stored under a folder named Experiment_1

```bash
Experiment_1/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```


### NaroNet

### BioInsights

### Demo
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment name | Example Image | Dataset link | Run in google colab |
| :-- | :-:| :-- | :-- |
| Discover tumoral differences between patient types (POLE gene mutated vs. POLE gene non-mutated) | <img src="https://github.com/djimenezsanchez/NaroNet/blob/main/images/example_endometrial_crop.png" title="example image fluo" width="320px" align="center">  | [Endometrial cancer tissue example (download Example_POLE.zip)](https://zenodo.org/record/4630664#.YFoGLa9KiUk). |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimenezsanchez/NaroNet/blob/main/examples/google_colab_example.ipynb?authuser=1) |


### Citation



