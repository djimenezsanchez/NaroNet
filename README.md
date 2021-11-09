# NaroNet: discovery of tumor microenvironment elements from highly multiplexed images.
***TL;DR:*** NaroNet is an end-to-end interpretable learning method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to classify subjects into predefined types. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding subject-level labels. See our [*paper*](https://arxiv.org/abs/2103.05385) for further description of NaroNet.  

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview.gif' />

© [CIMA Universidad de Navarra](https://cima.cun.es/en/research/research-programs/solid-tumors-program/research-group-preclinical-models-preclinical-tools-analysis) - This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes. 

## Index (the usage of this code is explained step by step) 
[Requirements and installation](#Requirements-and-installation) • [Preparing datasets](#Preparing-datasets) • [Preparing parameter configuration](#Preparing-parameter-configuration) • [Preprocessing](#Preprocessing) • [Patch Contrastive Learning](#Patch-Contrastive-Learning) • [NaroNet](#NaroNet) • [BioInsights](#BioInsights) • [Cite](#reference) • [Demo](#Demo) 

## Requirements and installation
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 4 on GPU server, and Nvidia P100, K80 GPUs on Google Cloud)

To install NaroNet we recommend creating a new [*anaconda*](https://www.anaconda.com/distribution/) environment with TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer). For GPU support, install the versions of CUDA that are compatible with TensorFlow's and Pytorch's versions.

Once inside the created environment, install pytorch-geometric where ${CUDA} and ${TORCH} should be replaced by the specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111, cu113) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0). Run the following commands in your console:
```sh
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

Install NaroNet downloading this repository or through pip:
```sh
pip install NaroNet
```

## Preparing datasets
Create the target folder (e.g., 'DATASET_DATA_DIR') with your image and subject-level information using the following folder structure:

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
In the 'Raw_Data/Images' folder we expect multiplex image data consisting of multi-page '.tiff' files with one channel/marker per page.
In the 'Raw_Data/Experiment_Information' two files are expected:
* Channels.txt contains per row the name of each marker/channel present in the multiplex image. In case the name of the row is 'None' it will be ignored and not loaded from the raw image.
```bash
Marker_1
Marker_2 
None
Marker_4    
```

* Image_Labels.xlsx contains the image names and their corresponding image-level labels. In column 'Image_Names' image names are specified. The next columns (e.g., 'Control vs. Treatment', 'Survival', etc.) specify image-level information, where 'None' means that the image is excluded from the experiment. In case more than one image is available per subject and you want to make it sure that images from the same subject don't go to different train/val/test splits, it is possible to add one column named "Subject_Names" specifying, for each image, the subject to whom it corresponds.

| Image_Names | Control vs. Treatment | Survival | 
| :-- | :-:| :-: |
| image_1.tiff | Control  | Poor |
| image_2.tiff | None | High |
| image_3.tiff | Treatment | High |
| ... | ... | ... |

* Patient_to_Image.xlsx (optional) can be utilized in case more than one image is available per subject and you want to merge them into one subject-graph. When images have the same subject identifier (e.g., 'Subject_Name') they will be joined into one disjoint graph. Please notice that when this file exists, you should change 'Image_Names' column in 'Image_Labels.xlsx' with the new subject names (e.g., change 'image_1.tiff' with 'subject_1').

| Image_Name | Subject_Name |
| :-- | :-:| 
| image_1.tiff | subject_1 |
| image_2.tiff | subject_1 | 
| image_3.tiff | subject_2 | 
| ... | ... | ... |

## Preparing parameter configuration
In the following sections (i.e., preprocessing, PCL, NaroNet, BioInsights) several parameters are required to be set. Although parameters will be explained in each section, all of them should be specified in the file named 'DatasetParameters.py', which is located in the folder 'NaroNet/src/utils'. Change it to your own configuration, where 'DATASET_DATA_DIR' is your target folder. Use examples as template: 
```python
def parameters(path, debug):
    if 'DATASET_DATA_DIR' in path:        
        args['param1'] = value1
	args['param2'] = value2
	...		
```

## Preprocessing
The firt step is to preprocess the image dataset and convert the raw image data to .npy files. To this end, 'NaroNet.preprocess_images' function is used. It uses the following parameters:
* `args['PCL_ZscoreNormalization']`: use z-score normalization so that each marker in the full cohort shows a mean of 0 and a standard deviation of 1. Default: True.
* `args['PCL_patch_size']`: size of the sides of a square image patch that will used as basic unit of interpretability. Default: 15 (pixels).

Once it is executed it will create the following green folder:

```diff
DATASET_DATA_DIR/
    ├── Raw_Data/
        ├── Images/
                ├── image_1.tiff
                ├── image_2.tiff
                └── ...
        └── ...
+   └── Patch_Contrastive_Learning/
+ 	└── Preprocessed_Images/
+    		├── Num_patches_perImage.csv
+		├── image_1.npy
+		├── image_2.npy		
+		└── ...
		
```

### Patch Contrastive Learning (PCL)
The goal of PCL in our pipeline is to convert each high-dimensional multiplex image of the cohort into a list of low-dimensional embedding vectors. To this end, each image is divided into patches -our basic units of representation containing one or two cells of the tissue-, and each patch is converted by the PCL module -a properly trained CNN- into a low-dimensional vector that embeds both the morphological and spectral information of the patch.

To this end, 'NaroNet.patch_contrastive_learning' function is used with the following parameters:
* `args['PCL_embedding_dimensions']`: size of the embedding vector generated for each image patch. Default: 256 (values)
* `args['PCL_batch_size']`: batch size of image patches used to train PCL's CNN. Example: 80 (image patches)
* `args['PCL_epochs']`: epochs to train PCL's CNN. Example: 500 (epochs) 
* `args['PCL_alpha_L']`: size ratio between image crops and augmented views used to train PCL's CNN. Default: 1.15. 
* `args['PCL_width_CNN']`: CNN's width multiplication factor. Default: 2.
* `args['PCL_depth_CNN']`: CNN's depth. Default: 101 (ResNet101).

When executed, PCL checks whether a trained CNN is already in a previously created folder named 'Model_Training_xxxx', where xxxx are random letters. In case the folder does not exist, PCL creates a new model, stores it in a new 'Model_Training_xxxx' folder, and trains it using the parameter configuration. To check whether the CNN has been trained successfully, check the 'Model_training_xxxx' folder and open the 'Contrast_accuracy_plot.png', where you should expect a final contrast accuracy value over 50%. 

Once the CNN is trained, execute again 'NaroNet.preprocess_images' to infer image patch representations from the whole dataset. Here, image patches are introduced in the CNN sequentially getting representation vectors back. For each image in the dataset, a npy data structure is created consisting of a matrix, where rows are patches, and columns are representation values. Here, the two first column values specify the x and y position of the patch in the image that will be later used to create a graph. In case Patient_to_Image.xlsx exists the npy structure will contain patches from more than one image.

Once executed you should expect the following folder structure, where Model_Training_xxxx is created during training, and Image_Patch_Representation during inference (in green):

```diff
DATASET_DATA_DIR/
    ├── Raw_Data/        
        └── ...
    └── Patch_Contrastive_Learning/
 	├── Preprocessed_Images/    		
		└── ...
+	├── Model_Training_xxxx/
+    		├── model.ckpt-0.index
+		├── model.ckpt-0-meta
+		├── model.ckpt-0.data-00000-of-00001
+		├── event.out.tfevents...
+		├── checkpoint
+		└── ...
+	└── Image_Patch_Representation/
+    		├── image_1.npy
+		├── image_2.npy
+		└── ...
```

## NaroNet
NaroNet inputs graphs of patches (stored in 'DATASET_DATA_DIR/Patch_Contrastive_Learning/Image_Patch_Representation') and subject-level labels (stored in 'DATASET_DATA_DIR/Raw_Data/Experiment_Information/Image_Labels.xlsx') to output subject's predictions from the abundance of learned phenotypes, neighborhoods, and areas. To this end, execute 'NaroNet.NaroNet.run_NaroNet' with the following parameters (most relevant parameters are shown, where additional ones are explained in DatasetParameters.py):

* `args['experiment_Label']`: Subject-level column labels from Image_Labels.xlsx to specify how you want to differentiate subjects. You can provide more than one column label name to perform multilabel classification. Example1: ['Survival'], Example2: ['Survival','Control vs. Treatment'].
* `args['epochs']`: Number of epochs to train NaroNet. Default: 20 (epochs).
* `args['weight_decay']`: Weight decay value. Default: 0.0001.
* `args['batch_size']`: Batch size. Default: 8 (subjects).
* `args['lr']`: Learning rate. Default: 0.001.
* `args['folds']`: Number of folds to perform cross-validation. Default: 10 (folds).
* `args['device']`: Specify whether to use cpu or gpu. Examples: 'cpu', 'cuda:0', 'cuda:1'.
* `args['clusters1']`: Number of phenotypes to be learned by NaroNet. Example: 10 (phenotypes). 
* `args['clusters2']`: Number of neighborhoods to be learned by NaroNet. Example: 11 (neighborhoods). 
* `args['clusters3']`: Number of areas to be learned by NaroNet. Example: 6 (areas). 

When executed, NaroNet selects the images included in the experiment and creates a graph of patches in the pytorch's format .pt. Next, k-fold-cross validation is carried out, training the model with 90% of the data and testing in the remaining 10%. See in green all folders created:

```diff
DATASET_DATA_DIR/
    ├── Raw_Data/        
        └── ...
    ├── Patch_Contrastive_Learning/		
	└── ...
+   └── NaroNet/		
+	├── Survival/ (experiment name example)
+		├── Subject_graphs/
+    			├── data_0_0.pt
+    			├── data_1_0.pt
+			└── ...
+		├── Cell_type_assignment/
+    			├── cluster_assignment_Index_0_ClustLvl_10.npy (phenotypes)
+    			├── cluster_assignment_Index_0_ClustLvl_11.npy (neighborhoods)
+    			├── cluster_assignment_Index_0_ClustLvl_6.npy (areas)
+    			├── cluster_assignment_Index_1_ClustLvl_10.npy (phenotypes)
+    			├── cluster_assignment_Index_1_ClustLvl_11.npy (neighborhoods)
+    			├── cluster_assignment_Index_1_ClustLvl_6.npy (areas)
+			└── ...
+		└── Cross_validation_results/
+			├── ROC_AUC_Survival.png
+			├── ConfusionMatrix_Survival.png
+			└── ...
```

## BioInsights
NaroNet's learned phenotypes, neighborhoods, and areas (stored in 'Cell_type_assignment'), can be analyzed _a posteriori_ by the BioInsights module. Here, elements of the tumor microenvironment are extracted, visualized, and associated to subject types. Execute 'NaroNet.NaroNet_dataset.get_BioInsights' with the same parameters as done in the NaroNet module to automatically generate the following folders:

* Cell_type_characterization: Contains heatmaps with the marker expression levels of phenotypes, neighborhoods, and areas. Contains examples 

```diff
DATASET_DATA_DIR/
    ├── Raw_Data/        
        └── ...
    ├── Patch_Contrastive_Learning/		
	└── ...
    ├── NaroNet/    	
	└── Survival/ (experiment name example)
		└── ...
+   └── BioInsights/
+    	└── Survival/ (experiment name example)
+		├── Phenotypes/
+
+ 		├── Neighborhoods/
+ 		└── Areas/

+		└── Cross_validation_results/
+			├── ROC_AUC_Survival.png
+			├── ConfusionMatrix_Survival.png
+			└── ...
```
## Demo
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment name | Example Image | Dataset link | Run in google colab |
| :-- | :-:| :-- | :-- |
| Discover tumoral differences between patient types (POLE gene mutated vs. POLE gene non-mutated) | <img src="https://github.com/djimenezsanchez/NaroNet/blob/main/images/example_endometrial_crop.png" title="example image fluo" width="320px" align="center">  | [Endometrial cancer tissue example (download Example_POLE.zip)](https://zenodo.org/record/4630664#.YFoGLa9KiUk). |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimenezsanchez/NaroNet/blob/main/examples/google_colab_example.ipynb?authuser=1) |


## Citation (please cite this paper in case our method or parts of it were helpful in your research)
@article{jimenez2021naronet,
  title={NaroNet: Discovery of tumor microenvironment elements from highly multiplexed images},
  author={Jim{\'e}nez-S{\'a}nchez, Daniel and Ariz, Mikel and Chang, Hang and Matias-Guiu, Xavier and de Andrea, Carlos E and Ortiz-de-Sol{\'o}rzano, Carlos},
  journal={arXiv preprint arXiv:2103.05385},
  year={2021}
}


