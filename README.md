# NaroNet: discovery of tumor microenvironment elements from highly multiplexed images.
***TL;DR:*** NaroNet is an end-to-end interpretable learning method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to predict subject-level labels. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding patient-level labels. See our [*paper*](https://arxiv.org/abs/2103.05385).  

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview.gif' />

© [Daniel Jiménez-Sánchez - CIMA Universidad de Navarra](https://cima.cun.es/investigacion/programas-investigacion/programa-tumores-solidos/grupo-investigacion-modelos-preclinicos-herramientas-analisis) - This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes. 

## Index (the usage of this code is explained step by step) 
[Requirements and installation](#Requirements-and-installation) • [Preparing datasets](#Preparing-datasets) • [Preparing parameter configuration](#Preparing-parameter-configuration) • [Preprocessing](#Preprocessing) • [Patch Contrastive Learning](#Patch-Contrastive-Learning) • [NaroNet](#NaroNet) • [BioInsights](#BioInsights) • [Cite](#reference) • [Demo](#Demo) 

## Requirements and installation
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

## Preparing datasets
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
In the 'Raw_Data/Images' folder we expect multiplex image data consisting of multi-page '.tiff' files with one channel/marker per page. Please notice that our method also works with RGB images. 
In the 'Raw_Data/Experiment_Information' two files are expected:
* Channels.txt contains per row the name of each marker/channel present in the multiplex image. In case the name of the row is 'None' it will be ignored and not loaded from the raw image.
```bash
Marker_1
Marker_2 
None
Marker_4    
```

* Image_Labels.xlsx contains the image-level labels. In column 'Image_Names' each row specifies one image. The next columns (e.g., 'Type_1', 'Type_2', etc.) specify image-level information, where 'None' means that there is no information available for this subject and therefore it has to be excluded from the experiment. In the case more than one image is available per subject, but you don't want to merge it in one subject graph it is possible to add one column named "Subject_Names", when this is included the method will make it sure that iamges from the same subject do nt go to different train/val/test splits.

| Image_Names | Type_1 | Type_2 | 
| :-- | :-:| :-: |
| image_1.tiff | A  | X |
| image_2.tiff | None | Y |
| image_3.tiff | B | Y |
| ... | ... | ... |

* Patient_to_Image.xlsx (optional) can be utilized in case more than one image is available per subject. When this file exists, our method creates subject graphs with more than one image on them. In column 'Image_Names' each row specifies one image. In 'Subject_Name' each row specifies one subject, meaning that when two or more rows have the same subject identifier it will join images into one disjoint graph. Please notice that when this file exists, you should change 'Image_Names' column in 'Image_Labels.xlsx' with the new subject names (e.g., change 'image_1.tiff' with 'subject_1').

| Image_Name | Subject_Name |
| :-- | :-:| 
| image_1.tiff | subject_1 |
| image_2.tiff | subject_1 | 
| image_3.tiff | subject_2 | 
| ... | ... | ... |

## Preparing parameter configuration
In the following sections (i.e., preprocessing, PCL, NaroNet, BioInsights) several parameters are defined and required to be set. Each parameter will be explained in each section, however all of them should be specified in the file named 'DatasetParameters.py', which is located in the folder 'NaroNet/src/utils'. Change it to your own configuration, where 'DATASET_DATA_DIR' is the name of the folder where the image data is stored. Use other examples as template. 
```python
def parameters(path, debug):
    if 'DATASET_DATA_DIR' in path:        
        args['param1'] = value1
	args['param2'] = value2
	...		
```

## Preprocessing
The firt step is to preprocess the image dataset and convert the raw image data to .npy files. To this end, 'NaroNet.preprocess_images' function is used. It uses the following parameters:
* `args['PCL_ZscoreNormalization']`: use z-score normalization so that each marker in the full cohort presents a mean 0 and a standard deviation of 1 (default: True)
* `args['PCL_patch_size']`: size of the square image patch that will be processed by the PCL module. Default: 15 (pixels)

Once it is executed it will create the following folder (in green):

```diff
DATASET_DATA_DIR/
    └── Raw_Data/
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
The goal of PCL in our pipeline is to convert each high-dimensional multiplex image of the cohort into a list of low-dimensional embedding vectors. To this end, each image is divided into patches -our basic units of representation containin one or two cells of the tissue-, and each patch is converted by the PCL module -a properly trained CNN- into a low-dimensional vector that embeds both the morphological and spectral information of the patch.

To this end, 'NaroNet.patch_contrastive_learning' function is used. It uses the following parameters:
* `args['PCL_embedding_dimensions']`: use z-score normalization so that each marker in the full cohort presents a mean 0 and a standard deviation of 1 (default: True)
* `args['PCL_batch_size']`: size of the square image patch that will be processed by the PCL module. Default: 15 (pixels)
* `args['PCL_epochs']`: size of the square image patch that will be processed by the PCL module. Default: 15 (pixels) 
* `args['PCL_patch_size']`: size of the square image patch that will be processed by the PCL module. Default: 15 (pixels) 
* `args['PCL_alpha_L']`: size of the square image patch that will be processed by the PCL module. Default: 15 (pixels)
* `args['PCL_width_CNN']`: size of the square image patch that will be processed by the PCL module. Default: 15 (pixels) 
* `args['PCL_depth_CNN']`: size of the square image patch that will be processed by the PCL module. Default: 15 (pixels) 

When executed, PCL checks whether a trained CNN is properly trained in the folder named 'Model_Training_xxxx', where xxxx are random letters. In case the folder does not exist, PCL creates a new model, stores it in a new 'Model_Training_xxxx' folder, and trains it using the parameter configuration. To check whether the CNN has been trained successfully, check the 'Model_training_xxxx' folder and open the 'Contrast_accuracy_plot.png', where you should expect a final contrast accuracy value over 50%. 

Once the CNN is trained, execute again 'NaroNet.preprocess_images' to infer image patch representations from the whole dataset. Here, image patches are introduced in the CNN sequentially getting representation vectors back. For each image in the dataset, a npy data structure is created consisting of a matrix, where rows are patches, and columns are representation values. Here, the two first column values specify the x and y position of the patch in the image. In case Patient_to_Image.xlsx exists the npy structure will contain patches from more than one image.

Once executed you should expect the following folder structure, where Model_Training_xxxx is created during training, and Image_Patch_Representation during inference (in green):

```diff
DATASET_DATA_DIR/
    └── Raw_Data/        
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
NaroNet inputs graphs of patches (stored in 'DATASET_DATA_DIR/Patch_Contrastive_Learning/Image_Patch_Representation') and subject-level labels (stored in 'DATASET_DATA_DIR/Raw_Data/Experiment_Information/Image_Labels.xlsx') and outputs subject-level predictions from learned phenotypes, neighborhoods, and areas. To this end, 


## BioInsights

## Demo
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment name | Example Image | Dataset link | Run in google colab |
| :-- | :-:| :-- | :-- |
| Discover tumoral differences between patient types (POLE gene mutated vs. POLE gene non-mutated) | <img src="https://github.com/djimenezsanchez/NaroNet/blob/main/images/example_endometrial_crop.png" title="example image fluo" width="320px" align="center">  | [Endometrial cancer tissue example (download Example_POLE.zip)](https://zenodo.org/record/4630664#.YFoGLa9KiUk). |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimenezsanchez/NaroNet/blob/main/examples/google_colab_example.ipynb?authuser=1) |


## Citation



