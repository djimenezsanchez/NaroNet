# NaroNet: discovery of tumor microenvironment elements from highly multiplexed images.
***Summary:*** NaroNet is an end-to-end interpretable learning method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to classify subjects into predefined types. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding subject-level labels. See our [*paper*](https://www.sciencedirect.com/science/article/pii/S1361841522000366) for further description of NaroNet.  

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview_big.png' />

© [Daniel Jiménez Sánchez - CIMA University of Navarra](https://cima.cun.es/en/research/research-programs/solid-tumors-program/research-group-preclinical-models-preclinical-tools-analysis) - This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes. 

## Index (the usage of this code is explained step by step) 
[Requirements and installation](#Requirements-and-installation) • [Preparing datasets](#Preparing-datasets) • [Preparing parameter configuration](#Preparing-parameter-configuration) • [Preprocessing](#Preprocessing) • [Patch Contrastive Learning](#Patch-Contrastive-Learning) • [NaroNet](#NaroNet) • [BioInsights](#BioInsights) • [Demo](#Demo)  • [Cite](#Citation) 

## Requirements and installation
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 4 on GPU server, and Nvidia P100, K80 GPUs on Google Cloud)

To install NaroNet we recommend creating a new [*anaconda*](https://www.anaconda.com/distribution/) environment with Pytorch (v.1.4.0 or newer). For GPU support, install the versions of CUDA that are compatible with Pytorch's versions.
```sh
conda create --name NaroNet python=3.8
```

Once inside the created environment, install pytorch and pytorch-geometric:
```sh
conda install pytorch torchvision torchaudio torchvision cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
```

Now you can install the following libraries using pip:
```sh
pip install hyperopt
pip install xlsxwriter
pip install matplotlib
pip install seaborn
pip install imgaug
sudo apt-get install python3-opencv 
pip install tensorboard
pip install openTSNE
pip install openpyxl
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
* Channels.txt contains per row the name of each marker/channel present in the multiplex image. In case the name of the row is 'None' it will be ignored and not loaded from the raw image. See example [file](https://github.com/djimenezsanchez/NaroNet/blob/main/examples/Channels.txt) or example below:
```bash
Marker_1
Marker_2 
None
Marker_4    
```

* Image_Labels.xlsx contains the image names and their corresponding image-level labels. In column 'Image_Names' image names are specified. The next columns (e.g., 'Control vs. Treatment', 'Survival', etc.) specify image-level information, where 'None' means that the image is excluded from the experiment. In case more than one image is available per subject and you want to make it sure that images from the same subject don't go to different train/val/test splits, it is possible to add one column named "Subject_Names" specifying, for each image, the subject to whom it corresponds. See example [file](https://github.com/djimenezsanchez/NaroNet/blob/main/examples/Image_Labels.xlsx) or example below:

| Image_Names | Control vs. Treatment | Survival | 
| :-- | :-:| :-: |
| image_1.tiff | Control  | Poor |
| image_2.tiff | None | High |
| image_3.tiff | Treatment | High |
| ... | ... | ... |

* Patient_to_Image.xlsx (optional) can be utilized in case more than one image is available per subject and you want to merge them into one subject-graph. When images have the same subject identifier (e.g., 'Subject_Name') they will be joined into one disjoint graph. Please notice that when this file exists, you should change 'Image_Names' column in 'Image_Labels.xlsx' with the new subject names (e.g., change 'image_1.tiff' with 'subject_1'). See example [file](https://github.com/djimenezsanchez/NaroNet/blob/main/examples/Patient_to_Image.xlsx) or example below:

| Image_Name | Subject_Name |
| :-- | :-:| 
| image_1.tiff | subject_1 |
| image_2.tiff | subject_1 | 
| image_3.tiff | subject_2 | 
| ... | ... | ... |

## Preparing parameter configuration
In the following sections (i.e., preprocessing, PCL, NaroNet, BioInsights) several parameters are required to be set. Although parameters will be explained in each section, all of them should be specified in the file named 'DatasetParameters.py', which is located in the folder 'NaroNet/src/utils'. Change it to your own configuration, where 'DATASET_DATA_DIR' is your target folder. See example [file](https://github.com/djimenezsanchez/NaroNet/blob/main/src/NaroNet/utils/DatasetParameters.py) or example below:
```python
def parameters(path, debug):
    if 'DATASET_DATA_DIR' in path:        
        args['param1'] = value1
	args['param2'] = value2
	...		
```

## Patch Contrastive Learning
The goal of PCL in our pipeline is to convert each high-dimensional multiplex image of the cohort into a list of low-dimensional embedding vectors. To this end, each image is divided into patches -our basic units of representation containing one or two cells of the tissue-, and each patch is converted by the PCL module -a properly trained CNN- into a low-dimensional vector that embeds both the morphological and spectral information of the patch.

To this end, 'NaroNet.patch_contrastive_learning' function is used with the following parameters:
* `args['PCL_Epochs']`: # Number of epochs to run PCL. Increase the numnber so that PCL achieves at least a 95% of top1 accuracy. Default: 100 (epochs).
* `args['PCL_N_Workers']`: # Number of workers to parallelize data loading. A higher number of workers will result in a faster execution but will require more RAM memory usage. Default: 1 (workers).
* `args['PCL_N_Crops_per_Image']`: # Number of image crops used to train the model in each iteration. Higher number of crops will learn more features from patches from one image but requires more RAM and GPU memory usage. Default: 100 (crops per image)
* `args['PCL_Batch_Size']`: # Number of images used per iteration. A higher batch size results in learning more features and doing it in a more stable way but will require more RAM and GPU memory usage. Default: 4 (images).    
* `args['PCL_eliminate_Black_Background']`:# Whether to eliminate from the training the black background. Use it for Multiplex fluorescence images. Default = True


When executed, PCL checks whether a CNN is already created in a folder named 'Model ':
- If the folder does not exist, PCL begins training a new model using the parameter configuration and creates a 'Model' folder in which to store it. 
- If the folder exists and there are less than 10 model checkpoints, PCL trains from a previously created model checkpoint (e.g., ‘checkpoint_0150.pth.tar’).
- If the folder exists and there are 10 checkpoints within it, PCL does not train.

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

* Cell_type_characterization: Contains heatmaps with the marker expression levels of phenotypes, neighborhoods, and areas. Also contains examples of patches assigned to the phenotypes, neighborhoods and areas.

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Figura_Phentypes.gif' />

* Cell_type_abundance: Contains heatmaps with the abundance of phenotypes, neighborhoods, and areas per subject. This information is then used to perform the differential TME composition analysis.

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Figura_ConfidencePredictions_ithub.gif' />

* Differential_abundance_analysis: provides information about the differential TME composition analysis (p-values specify predictive power of TMEs). It also provides statistical tests showing if found TMEs are cohort-differenting. Examples of predicted subjects are stored in the folder 'Locate_TME_in_image'.

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Figura_Areas_github.gif' />


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
+		├── Cell_type_characterization/
+			└── ...
+		├── Cell_type_abundance/
+			└── ...
+		├── Differential_abundance_analysis/
+			└── ...
+ 		└── Locate_TME_in_image/
+			└── ...

```
## Demo
We provide an example workflow via Jupyter notebook that illustrate how this package can be used.

| Experiment name | Example Image | Dataset link | Run in google colab |
| :-- | :-:| :-- | :-- |
| Discover tumoral differences between patient types (POLE gene mutated vs. POLE gene non-mutated) | <img src="https://github.com/djimenezsanchez/NaroNet/blob/main/images/example_endometrial_crop.png" title="example image fluo" width="320px" align="center">  | [Endometrial cancer tissue example (download Example_POLE.zip)](https://zenodo.org/record/4630664#.YFoGLa9KiUk). |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djimenezsanchez/NaroNet/blob/main/examples/google_colab_example.ipynb?authuser=1) |


## Citation
Please cite this paper in case our method or parts of it were helpful in your work.
```diff
@article{jimenez2021naronet,
  title={NaroNet: Discovery of tumor microenvironment elements from highly multiplexed images},
  author={Jiménez-Sánchez, Daniel and Ariz, Mikel and Chang, Hang and Matias-Guiu, Xavier and de Andrea, Carlos E and Ortiz-de-Solórzano, Carlos},
  journal={Medical image analysis, vol. 78 102384.},
  year={2022}
}
```


