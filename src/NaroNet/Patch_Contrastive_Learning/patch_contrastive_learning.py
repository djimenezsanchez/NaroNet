import yaml
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import random
import os
import numpy as np
from imgaug import augmenters as iaa
import torch.nn.functional as F
import xlrd
import pandas as pd
import string
from tensorboard.backend.event_processing import event_accumulator
from subprocess import Popen,PIPE
import ctypes 

# SIMCLR_ASPAPER
import time	
import threading
import tensorflow.compat.v1 as tf    
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util   
tf.get_logger().setLevel('ERROR')
tf.debugging.set_log_device_placement(True)
from NaroNet.Patch_Contrastive_Learning.simclr import resnet
import NaroNet.Patch_Contrastive_Learning.simclr.model_util as model_util
import NaroNet.Patch_Contrastive_Learning.simclr.model as model_lib
import NaroNet.Patch_Contrastive_Learning.simclr.data as data_lib
from absl import flags
# flags.DEFINE_string('f', path+'Patch_Contrastive_Learning/','path.')
from absl import app
import math
import shutil
import csv
from tqdm import tqdm

def define_flags(path,params,n_images):    

    flags_dict = flags.FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        if not any([keys=='verbosity',keys=='logger_levels',keys=='only_check_args',
        keys=='pdb_post_mortem',keys=='run_with_pdb', keys=='run_with_profiling',keys=='profile_file',
        keys=='logtostderr',keys=='stderrthreshold',keys=='alsologtostderr',keys=='showprefixforinfo',keys=='f']):
            flags.FLAGS.__delattr__(keys)
    
    flags.DEFINE_string(
        'path', path+'Patch_Contrastive_Learning/',
        'path.')

    flags.DEFINE_float(
        'PCL_alpha_L', params['PCL_alpha_L'],
        'PCL_alpha_L.')    

    flags.DEFINE_integer(
        'n_images', n_images,
        'n_images.')
    
    flags.DEFINE_integer(
        'n_images_iteration', 8,
        'n_images_iteration.')

    flags.DEFINE_string(
        'device', params['device'],
        'device.')

    flags.DEFINE_integer(
        'PCL_embedding_dimensions', params['PCL_embedding_dimensions'],
        'PCL_embedding_dimensions.')

    flags.DEFINE_string(
        'output_path', path+'Patch_Contrastive_Learning',
        'output_path.')

    flags.DEFINE_float(
        'learning_rate', 0.5,
        'Initial learning rate per batch size of 256.')    

    flags.DEFINE_float(
        'warmup_epochs', 10,
        'Number of epochs of warmup.')

    flags.DEFINE_float(
        'weight_decay', 1e-6,
        'Amount of weight decay to use.')

    flags.DEFINE_float(
        'batch_norm_decay', 0.9,
        'Batch norm decay parameter.')

    flags.DEFINE_integer(
        'train_batch_size', params['PCL_batch_size'],
        'Batch size for training.')

    flags.DEFINE_string(
        'train_split', 'train',
        'Split for training.')

    flags.DEFINE_integer(
        'PCL_epochs', params['PCL_epochs'],
        'Number of epochs to train for.')

    flags.DEFINE_integer(
        'train_steps', 0,
        'Number of steps to train for. If provided, overrides train_epochs.')

    flags.DEFINE_integer(
        'eval_batch_size', params['PCL_batch_size'],
        'Batch size for eval.')

    flags.DEFINE_integer(
        'train_summary_steps', 100,
        'Steps before saving training summaries. If 0, will not save.')

    flags.DEFINE_integer(
        'checkpoint_epochs', max(int(params['PCL_epochs']/10),1),
        'Number of epochs between checkpoints/summaries.')

    flags.DEFINE_integer(
        'checkpoint_steps', 0,
        'Number of steps between checkpoints/summaries. If provided, overrides '
        'checkpoint_epochs.')

    flags.DEFINE_string(
        'eval_split', 'test',
        'Split for evaluation.')

    flags.DEFINE_string(
        'dataset', 'dataset',
        'Name of a dataset.')

    flags.DEFINE_bool(
        'cache_dataset', False,
        'Whether to cache the entire dataset in memory. If the dataset is '
        'ImageNet, this is a very bad idea, but for smaller datasets it can '
        'improve performance.')

    flags.DEFINE_enum(
        'mode', 'train', ['train', 'eval', 'train_then_eval'],
        'Whether to perform training or evaluation.')

    flags.DEFINE_enum(
        'train_mode', 'pretrain', ['pretrain', 'finetune'],
        'The train mode controls different objectives and trainable components.')

    flags.DEFINE_string(
        'checkpoint', None,
        'Loading from the given checkpoint for continued training or fine-tuning.')

    flags.DEFINE_string(
        'variable_schema', '?!global_step',
        'This defines whether some variable from the checkpoint should be loaded.')

    flags.DEFINE_bool(
        'zero_init_logits_layer', False,
        'If True, zero initialize layers after avg_pool for supervised learning.')

    flags.DEFINE_integer(
        'fine_tune_after_block', -1,
        'The layers after which block that we will fine-tune. -1 means fine-tuning '
        'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
        'just the linera head.')

    flags.DEFINE_string(
        'master', None,
        'Address/name of the TensorFlow master to use. By default, use an '
        'in-process master.')

    flags.DEFINE_string(
        'data_dir', None,
        'Directory where dataset is stored.')

    flags.DEFINE_bool(
        'use_tpu', False,
        'Whether to run on TPU.')

    tf.flags.DEFINE_string(
        'tpu_name', None,
        'The Cloud TPU to use for training. This should be either the name '
        'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
        'url.')

    tf.flags.DEFINE_string(
        'tpu_zone', None,
        '[Optional] GCE zone where the Cloud TPU is located in. If not '
        'specified, we will attempt to automatically detect the GCE project from '
        'metadata.')

    tf.flags.DEFINE_string(
        'gcp_project', None,
        '[Optional] Project name for the Cloud TPU-enabled project. If not '
        'specified, we will attempt to automatically detect the GCE project from '
        'metadata.')

    flags.DEFINE_enum(
        'optimizer', 'lars', ['momentum', 'adam', 'lars'],
        'Optimizer to use.')

    flags.DEFINE_float(
        'momentum', 0.9,
        'Momentum parameter.')

    flags.DEFINE_string(
        'eval_name', None,
        'Name for eval.')

    flags.DEFINE_integer(
        'keep_checkpoint_max', 10,
        'Maximum number of checkpoints to keep.')

    flags.DEFINE_integer(
        'keep_hub_module_max', 1,
        'Maximum number of Hub modules to keep.')

    flags.DEFINE_float(
        'temperature', 0.5,
        'Temperature parameter for contrastive loss.')

    flags.DEFINE_boolean(
        'hidden_norm', True,
        'Temperature parameter for contrastive loss.')

    flags.DEFINE_enum(
        'head_proj_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
        'How the head projection is done.')

    flags.DEFINE_integer(
        'head_proj_dim', 128,
        'Number of head projection dimension.')

    flags.DEFINE_integer(
        'num_nlh_layers', 1,
        'Number of non-linear head layers.')

    flags.DEFINE_boolean(
        'global_bn', True,
        'Whether to aggregate BN statistics across distributed cores.')

    flags.DEFINE_integer(
        'width_multiplier', params['PCL_width_CNN'],
        'Multiplier to change width of network.')

    flags.DEFINE_integer(
        'resnet_depth', params['PCL_depth_CNN'], # 50 or 101
        'Depth of ResNet.')

    flags.DEFINE_integer(
        'patch_size', params['PCL_patch_size'],
        'Input image size.')

    flags.DEFINE_float(
        'color_jitter_strength', 0.5,
        'The strength of color jittering.')

    flags.DEFINE_boolean(
        'use_blur', False,
        'Whether or not to use Gaussian blur for augmentation during pretraining.')

class DatasetGenerator(Dataset):
    def __new__(cls):
        files = [i for i in os.listdir(flags.FLAGS.path) if 'patches.npy' in i]        
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float32,
            # output_shapes=(flags.FLAGS.patch_size,flags.FLAGS.patch_size),
            args=(files,)
        )
    
    def _generator(files):
        images=[]
        for indx, filee in enumerate(files):
            image = np.load(self.path+filee).squeeze()
            xIn,yIn = np.random.randint(0,data.shape[0]-int(flags.FLAGS.patch_size*2)-1), np.random.randint(0,data.shape[1]-int(flags.FLAGS.patch_size*2)-1)
            images.append(image[xIn:xIn+self.patch_size*2,yIn:yIn+self.patch_size*2,:])
        return numpy.stack(images)

class MyCustomDataset(Dataset):    
    '''
    This class generates a dataset that is used to load the whole experiment 
    to train the patch contrastive learning framework.
    '''
    
    def __init__(self, path, n_images, patch_size):                
        '''
        Initialize the object
        path: 'string' that specifies the folder where the images are located
        n_images: 'int' that specifies the number of images available to train the model
        patch_size: 'int' that specifies the number of pixels a patch is formed by.
        '''
        
        # Initialize the dataset
        self.n_images = n_images
        self.path = path+'Preprocessed_Images/'
        self.patch_size=patch_size
        self.indexImageTEST = 0   
        self.PCL_alpha_L = flags.FLAGS.PCL_alpha_L
        
        # Generate a list that specifies the number of patches present in each image.
        self.num_patches_inImage={}        
        with open(self.path+'Num_patches_perImage.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.num_patches_inImage[row[0]]=int(row[1])

    def get_patches_from_image(self,index):
        '''
        Loads an image and crop the image into several patches
        index: (int) that specifies the number of the image that will be opened
        '''
        
        # Load image
        data =  np.load(self.path+self.files[index])
        data = data.squeeze()

        # Obtain a crop of size Patch_size*alpha_L
        images=[]
        for _ in range(int(flags.FLAGS.train_batch_size/flags.FLAGS.n_images_iteration)):
            # Get random x,y positions within the image and append the crop
            xIn,yIn = np.random.randint(0,data.shape[0]-int(self.patch_size*self.PCL_alpha_L)-1), np.random.randint(0,data.shape[1]-int(self.patch_size*self.PCL_alpha_L)-1)
            images.append(data[xIn:xIn+int(self.patch_size*self.PCL_alpha_L),yIn:yIn+int(self.patch_size*self.PCL_alpha_L),:])

        return np.stack(images) 

    def __getitem__(self, index):               
        seq1 = iaa.Sequential([
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5), # Drop 2% of all pixels, 15% of the original size,in 50% of all patches channeld
            iaa.Add((-5, 5), per_channel=0.5), # Add noise per channel...
            iaa.Rotate((-180, 180)), # Rotate the patch...
            iaa.GaussianBlur(sigma=(0, 3.0)) # blur patches with a sigma of 0 to 3.0
        ])
        seq2 = iaa.Sequential([
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5), # Drop 2% of all pixels, 15% of the original size,in 50% of all images channeld
            iaa.Add((-5, 5), per_channel=0.5), # Add noise per channel...
            iaa.Rotate((-180, 180)), # Rotate the patch...
            iaa.GaussianBlur(sigma=(0, 3.0)) # blur patches with a sigma of 0 to 3.0
        ])

        data =  np.load(self.path+self.files[index])
        data = data.squeeze()

        # data crop
        xIn,yIn = np.random.randint(self.patch_size,data.shape[0]-int(self.patch_size*2)-1), np.random.randint(self.patch_size,data.shape[1]-int(self.patch_size*2)-1)
        xIn2, yIn2 =xIn+np.random.randint(-self.patch_size,self.patch_size), yIn+np.random.randint(-self.patch_size,self.patch_size) 
        xi = data[xIn:xIn+self.patch_size,yIn:yIn+self.patch_size,:]
        xj = data[xIn2:xIn2+self.patch_size,yIn2:yIn2+self.patch_size,:]

        # Color_Distortion...->...Sobel...->...Gaussian_Noise...->...Rotate
        xi = seq1(images=xi)
        xj = seq2(images=xj)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return xi, xj

    def loadOneImage(self, indexImage):
        data = np.load(self.path+self.files[indexImage])
        return data.squeeze()

    def save_test_info(self, files_names,patches_numbers,patches_position,patches_marker_mean):
        self.files_names = files_names
        self.patches_numbers = patches_numbers
        self.patches_position = patches_position
        self.patches_marker_mean = patches_marker_mean

    def getitem_TEST(self, image, indexPatch):               
        # data crop
        num_xGrids = int(image.shape[0]/self.patch_size)
        num_yGrids = int(image.shape[1]/self.patch_size)
        xIn = int(np.floor(indexPatch/num_yGrids)*self.patch_size)
        yIn = int(np.remainder(indexPatch,num_yGrids)*self.patch_size)      
        xi = image[xIn:xIn+self.patch_size,yIn:yIn+self.patch_size,:]        
        # Obtain Centroid        
        xIn = xIn+self.patch_size/2
        yIn = yIn+self.patch_size/2
        return xi, np.array([xIn,yIn],dtype='float32'), xi.mean((0,1))

    def __len__(self):
        return self.n_samples # of how many data(images?) you have
    
    def shuffle(self):
        random.shuffle(self.files)

def perform_evaluation_v3(estimator, training, input_fn, dataset, eval_steps, model, num_classes, checkpoint_path=None):
    """Perform evaluation.
    Args:
        estimator: TPUEstimator instance.
        input_fn: Input function for estimator.
        eval_steps: Number of steps for evaluation.
        model: Instance of transfer_learning.models.Model.
        num_classes: Number of classes to build model for.
        checkpoint_path: Path of checkpoint to evaluate.
    Returns:
        result: A Dict of metrics and their values.
    """
    if not checkpoint_path:
        checkpoint_path = estimator.latest_checkpoint()
    # result = estimator.evaluate(
    #     input_fn, eval_steps, checkpoint_path=checkpoint_path,
    #     name=flags.FLAGS.eval_name)
    # result = estimator.predict(
    #     input_fn)#, eval_steps, checkpoint_path=checkpoint_path,
    #    # name=flags.FLAGS.eval_name)

    # Create directory if it doesnt exist
    folder = dataset.path[:-11]+'OriginalSuperPatch/'
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("Directory " , folder ,  " Created ")
    else:    
        print("Directory " , folder ,  " already exists")

    # Eliminate files located in folder /OriginalSuperPatch
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))    
    
    if 'ZuriBasel' in dataset.ExperimentFolder:
        # with open(os.path.dirname(dataset.ExperimentFolder[:-1])+'/Raw/Basel_PatientMetadata.csv', newline='') as f:
        #     reader = csv.reader(f)
        #     Basel = list(reader)
        # workbook = xlrd.open_workbook(os.path.dirname(dataset.ExperimentFolder[:-1])+'/Raw/Basel_PatientMetadata.csv')
        # Basel = workbook.sheet_by_index(0)  
        BaselData = pd.read_excel(os.path.dirname(dataset.ExperimentFolder[:-1])+'/Raw/Basel_PatientMetadata.xlsx')
        ZuriData = pd.read_excel(os.path.dirname(dataset.ExperimentFolder[:-1])+'/Raw/Zuri_PatientMetadata.xlsx')

    elif 'Endometrial_LowGrade' in dataset.ExperimentFolder:
        EndoData = pd.read_excel(dataset.ExperimentFolder[:-12]+'/Raw/patient2Image.xlsx')
        OrderedListOfFiles = []
        for eD in EndoData['Name']:
            for f in dataset.files:
                if eD==f[:-12]:
                    OrderedListOfFiles.append(f)
        dataset.files = OrderedListOfFiles

    # Obtain and Save PCL representations  
    viz_rep = {}
    viz_rep_Saved = [True for _ in dataset.files]
        
    # Iterate over images
    for n_file in range(len(dataset.files)): 
        result = estimator.predict(input_fn(training, flags.FLAGS.train_batch_size, dataset, flags.FLAGS.patch_size,n_file), checkpoint_path=checkpoint_path)
        for first ,r in enumerate(result):
            # Obtain the number of patches per image on the first iteration
            if first==0:
                num_patches_per_image = []
                for idx, flnm in enumerate(dataset.files_names):
                    num_patches_per_image.append(dataset.num_patches_inImage[dataset.files.index(flnm)])    
            
            # File Name and numpy array to be saved
            index_patch = int(r['index'])
            index_image = dataset.files.index(dataset.files_names[index_patch])
            file_name = 'SuperPatch'+dataset.files_names[index_patch][5:-16]+'.npy'
            SuperPatch = np.expand_dims(np.concatenate((dataset.patches_position[index_patch],dataset.patches_marker_mean[index_patch],r['hiddens'])),axis=0)
            
            # Check if we already started filling data in this image
            if str(index_image) in viz_rep:
                if viz_rep_Saved[index_image]:
                    viz_rep[str(index_image)] = np.concatenate((viz_rep[str(index_image)],SuperPatch),axis=0)
                else:
                    break
            else:
                viz_rep[str(index_image)] = SuperPatch

            # Save Numpy array if data collection finished.
            if viz_rep[str(index_image)].shape[0]==num_patches_per_image[index_patch]:
                # Several Images per patient:
                if 'ZuriBasel' in dataset.ExperimentFolder:
                    # Obtain PID of patient
                    try:
                        if 'Basel' in dataset.files[n_file]:
                            index = list(BaselData['FileName_FullStack']).index(dataset.files[n_file][:-17]+'.tiff')
                            PID = BaselData['PID'][index]
                        elif 'ZTMA' in dataset.files[n_file]:
                            index = list(ZuriData['FileName_FullStack']).index(dataset.files[n_file][:-17]+'.tiff')
                            PID = ZuriData['PID'][index]
                        # Save Image of Patient or concatenate with previous images.
                        if os.path.exists(folder+str(PID)+".npy"):
                            PreviousImage = np.load(folder+str(PID)+".npy")
                            viz_rep[str(index_image)][:,[0,1]] = viz_rep[str(index_image)][:,[0,1]] + PreviousImage[:,[0,1]].max() + 100
                            ConcatenatedMatrix = np.concatenate((PreviousImage,viz_rep[str(index_image)]),axis=0)
                            np.save(folder+str(PID)+".npy",ConcatenatedMatrix)
                        else:
                            np.save(folder+str(PID)+".npy",viz_rep[str(index_image)])   
                    except:
                        break # This image doesnt correspond to any patient, break bucle and go to the next image
                    viz_rep[str(index_image)]=None
                    viz_rep_Saved[index_image]=False
                
                elif 'Endometrial_LowGrade' in dataset.ExperimentFolder:
                    index = list(EndoData['Name']).index(dataset.files[n_file][:-16]+'.tif')
                    PID = EndoData['ID'][index]
                    # Save Image of Patient or concatenate with previous images.
                    if os.path.exists(folder+str(PID)+".npy"):
                        PreviousImage = np.load(folder+str(PID)+".npy")                        
                        viz_rep[str(index_image)][:,[0,1]] = viz_rep[str(index_image)][:,[0,1]] + PreviousImage[:,[0,1]].max() + 100
                        ConcatenatedMatrix = np.concatenate((PreviousImage,viz_rep[str(index_image)]),axis=0)
                        np.save(folder+str(PID)+".npy",ConcatenatedMatrix)
                    else:
                        np.save(folder+str(PID)+".npy",viz_rep[str(index_image)])  
                    # Clean for further work
                    viz_rep[str(index_image)]=None
                    viz_rep_Saved[index_image]=False

                else:
                    np.save(folder+file_name,viz_rep[str(index_image)])   
                    viz_rep[str(index_image)]=None
                    viz_rep_Saved[index_image]=False

                # elif 'AlfonsoCalvo' in dataset.ExperimentFolder:
                #     np.save(folder+file_name,viz_rep[str(index_image)])   
                #     viz_rep[str(index_image)]=None
                #     viz_rep_Saved[index_image]=False
                


    # # Record results as JSON.
    # result_json_path = os.path.join(flags.FLAGS.model_dir, 'result.json')
    # with tf.io.gfile.GFile(result_json_path, 'w') as f:
    #     json.dump({k: float(v) for k, v in result.items()}, f)
    # result_json_path = os.path.join(
    #     flags.FLAGS.model_dir, 'result_%d.json'%result['global_step'])
    # with tf.io.gfile.GFile(result_json_path, 'w') as f:
    #     json.dump({k: float(v) for k, v in result.items()}, f)
    # flag_json_path = os.path.join(flags.FLAGS.model_dir, 'flags.json')
    # with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    #     json.dump(flags.FLAGS.flag_values_dict(), f)

    # # Save Hub module.
    # build_hub_module(model, num_classes,
    #                 global_step=result['global_step'],
    #                 checkpoint_path=checkpoint_path)

    return result

def perform_evaluation_v4(estimator, input_fn, dataset, checkpoint_path=None):
    """
    Infer visual representation for each patch.
    estimator: TPUEstimator instance.
    input_fn: Input function for estimator.
    dataset: instance of image loader.
    checkpoint_path: Path of checkpoint to evaluate.
    """

    # Obtain the path of the trained model
    checkpoint_path = estimator.latest_checkpoint()

    # Create directory if it doesnt exist
    folder = dataset.path[:-20]+'Image_Patch_Representation/'
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Obtain image filenames.
    dataset.files = [i for i in dataset.num_patches_inImage]

    # Check If the image patches were already Inferred
    if len(os.listdir(folder))>0:
        with tqdm(total=len(dataset.files),ascii=True, desc='PCL: Infer image patches') as bar_acc:
            bar_acc.update(len(dataset.files))
        return
    elif len(os.listdir(folder))>0:
        shutil.rmtree(folder)
        os.mkdir(folder)
        

    # In case there are more than one image per patient.
    patient_to_image = [i for i in os.listdir(dataset.path[:-47]+'Raw_Data/Experiment_Information/') if 'Patient_to_Image.xlsx'==i]
    if len(patient_to_image)>0:
        patient_to_image_excel = pd.read_excel(dataset.path[:-47]+'Raw_Data/Experiment_Information/'+patient_to_image[0])
        OrderedListOfFiles = []
        
        # Obtain list of image names and order them. This is done to make mosaics with the same order.
        for eD in patient_to_image_excel['Image_Name']:
            for f in dataset.files:
                if '.'.join(eD.split('.')[:-1])=='.'.join(f.split('.')[:-1]):
                    OrderedListOfFiles.append(f)
        dataset.files = OrderedListOfFiles
        dataset.format_name = '.'+patient_to_image_excel['Image_Name'][0].split('.')[-1]

    # Obtain and save PCL representations
    viz_rep = {}
    viz_rep_Saved = [True for _ in dataset.files]
        
    # Iterate over images
    for n_file in tqdm(range(len(dataset.files)),ascii=True,desc='PCL: Infer image patches'): 
        
        # Infer visual representation of an image
        result = estimator.predict(input_fn(False, flags.FLAGS.train_batch_size, dataset, flags.FLAGS.patch_size,n_file), checkpoint_path=checkpoint_path)

        # Obtain visual representation per patch
        for first ,r in enumerate(result):
                        
            # Obtain the number of patches per image on the first iteration
            if first==0:
                num_patches = []
                for idx, flnm in enumerate(dataset.files_names):
                    num_patches.append(dataset.num_patches_inImage[flnm])    
            
            # File Name and numpy array to be saved
            index_patch = int(r['index'])
            index_image = dataset.files.index(dataset.files_names[index_patch])
            file_name = dataset.files_names[index_patch]
            SuperPatch = np.expand_dims(np.concatenate((dataset.patches_position[index_patch],dataset.patches_marker_mean[index_patch],r['hiddens'])),axis=0)
            
            # Check if we already started filling data in this image
            if str(index_image) in viz_rep:
                if viz_rep_Saved[index_image]:
                    viz_rep[str(index_image)] = np.concatenate((viz_rep[str(index_image)],SuperPatch),axis=0)
                else:
                    break
            else:
                viz_rep[str(index_image)] = SuperPatch

            # Save Numpy array if data collection finished.
            if viz_rep[str(index_image)].shape[0]==num_patches[index_patch]:
                
                if len(patient_to_image)>0:                
                    index = list(patient_to_image_excel['Image_Name']).index('.'.join(dataset.files[n_file].split('.')[:-1])+dataset.format_name)
                    PID = patient_to_image_excel['Subject_Name'][index]
                    
                    # Save Image of Patient or concatenate with previous images.
                    if os.path.exists(folder+str(PID)+".npy"):
                        PreviousImage = np.load(folder+str(PID)+".npy")                        
                        viz_rep[str(index_image)][:,[0,1]] = viz_rep[str(index_image)][:,[0,1]] + PreviousImage[:,[0,1]].max() +  flags.FLAGS.patch_size*2
                        ConcatenatedMatrix = np.concatenate((PreviousImage,viz_rep[str(index_image)]),axis=0)
                        np.save(folder+str(PID)+".npy",ConcatenatedMatrix)
                    else:
                        np.save(folder+str(PID)+".npy",viz_rep[str(index_image)])  
                    
                    # Clean for further work
                    viz_rep[str(index_image)]=None
                    viz_rep_Saved[index_image]=False
                else:
                    # Save data
                    np.save(folder+file_name,viz_rep[str(index_image)])  
                   
                    # Clean for further work
                    viz_rep[str(index_image)]=None
                    viz_rep_Saved[index_image]=False

        # else:
        #     np.save(folder+file_name,viz_rep[str(index_image)])   
        #     viz_rep[str(index_image)]=None
        #     viz_rep_Saved[index_image]=False



def my_summary_iterator(summary_path):
    for r in tf_record.tf_record_iterator(summary_path):
        yield event_pb2.Event.FromString(r)

def training_or_inference(total_steps):
    '''
    Decide whether we should continue training the model, or contrarily stop training the model and start the inference
    total_steps: (int) that specifies the total number of steps that are calculated with respect the number of epochs
    '''

    # Load the existing model checkpoints.
    model_name = [l for l in os.listdir(flags.FLAGS.path) if 'Model_Training' in l][0]
    chekpointfiles = [l for l in os.listdir(flags.FLAGS.path+model_name) if ('events' in l) and ('.v2' in l)]
    if len(chekpointfiles)==0:
        return True, flags.FLAGS.path+model_name
    checkpoint_file = sorted(chekpointfiles, key=lambda t: -os.stat(flags.FLAGS.path+model_name+'/'+t).st_mtime)[0]
    if len(checkpoint_file)>0:
        summary_path = flags.FLAGS.path+model_name+'/'+checkpoint_file
        
        # Extract train contrast acc
        steps = []
        train_contrast_acc = []
        for event in my_summary_iterator(summary_path):
            for value in event.summary.value:
                if value.tag == 'train_contrast_acc':
                    t = tensor_util.MakeNdarray(value.tensor)
                    steps.append(event.step)
                    train_contrast_acc.append(t)                 
        train_contrast_acc = np.stack(train_contrast_acc)
        steps = np.array(steps)

        # Calculate the size for the average moving window.
        n_step_between_vals = sum(np.diff(steps))/(len(steps)-1)
        window_size = (steps.shape[0]*0.15)/n_step_between_vals
        train_contrast_acc_ws = np.convolve(train_contrast_acc, np.ones(max(int(window_size),1)), 'valid') / max(int(window_size),1)        

        # Show the model's contrast accuracy
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.plot(steps,train_contrast_acc*100, linewidth=3, label='Real values',color='blue')
        plt.plot(steps[len(train_contrast_acc)-len(train_contrast_acc_ws):], train_contrast_acc_ws*100, label='Average',color='orange')
        plt.xlabel('Steps')
        plt.ylabel('Contrast accuracy (%)')
        plt.title('Contrast Accuracy (last value: '+str(np.round(train_contrast_acc_ws[-1]*100,2))+'%)')
        plt.legend()
        plt.savefig(flags.FLAGS.path+model_name+'/Contrast_accuracy_plot.png',dpi=600)

        if total_steps-steps[-1]<total_steps*0.05:
            with tqdm(total=total_steps, ascii=True, desc="PCL: Train CNN (steps)") as bar_step:
                with tqdm(total= 1000,ascii=True, desc="PCL: Contrast Accuracy (per mille)") as bar_acc:
                    bar_step.update(steps[-2])
                    bar_acc.update(int(np.round(train_contrast_acc_ws[-1]*1000,2)))
            return False, flags.FLAGS.path+model_name # The model finalized to train.
        else:            
            return True, flags.FLAGS.path+model_name # The model hasn't end to learn
    else:
        return True, None

def load_PCL_model(epoch_steps,folder_model_dir,num_patches_epoch,dataset,checkpoint_dir):
    '''
    Generate CNN Model
    '''

    # Generate the model
    resnet.BATCH_NORM_DECAY = flags.FLAGS.batch_norm_decay
    model = resnet.resnet_v1(
        resnet_depth=flags.FLAGS.resnet_depth,
        width_multiplier=flags.FLAGS.width_multiplier,
        cifar_stem=True if flags.FLAGS.patch_size<64 else False,
        dataset=dataset)

    # Number of times the model is going to be saved
    checkpoint_steps = (flags.FLAGS.checkpoint_epochs * epoch_steps)#)# (flags.FLAGS.checkpoint_steps)# or 

    cluster = None

    default_eval_mode = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V1
    sliced_eval_mode = tf.estimator.tpu.InputPipelineConfig.SLICED
    run_config = tf.estimator.tpu.RunConfig(
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=checkpoint_steps,
            eval_training_input_configuration=sliced_eval_mode if flags.FLAGS.use_tpu else default_eval_mode),
        model_dir=folder_model_dir,
        save_summary_steps=checkpoint_steps,
        save_checkpoints_steps=checkpoint_steps,
        keep_checkpoint_max=flags.FLAGS.keep_checkpoint_max,
        master=flags.FLAGS.master,
        cluster=cluster)
        # session_config=tf.ConfigProto(device_count={'GPU': 0}))#,
                            # inter_op_parallelism_threads=10,
                            # intra_op_parallelism_threads=10)))
    if checkpoint_dir==None:
        estimator = tf.estimator.tpu.TPUEstimator(
            model_lib.build_model_fn(model, 5, num_patches_epoch, dataset, folder_model_dir),
            config=run_config,
            train_batch_size=flags.FLAGS.train_batch_size,
            eval_batch_size=flags.FLAGS.train_batch_size,
            predict_batch_size=flags.FLAGS.train_batch_size,
            use_tpu=flags.FLAGS.use_tpu)
    else:
        estimator = tf.estimator.tpu.TPUEstimator(
            model_lib.build_model_fn(model, 5, num_patches_epoch, dataset, folder_model_dir),
            config=run_config,
            train_batch_size=flags.FLAGS.train_batch_size,
            eval_batch_size=flags.FLAGS.train_batch_size,
            predict_batch_size=flags.FLAGS.train_batch_size,
            use_tpu=flags.FLAGS.use_tpu,
            warm_start_from=checkpoint_dir)
    return estimator

class progress_bar(threading.Thread): 
    def __init__(self,train_steps, *args, **kwargs): 
        super(progress_bar, self).__init__(*args, **kwargs)         
        self.train_steps = train_steps
        self._stop = threading.Event() 

    def extract_train_contrast_acc(self,summary_path):
        # Extract train contrast acc
        steps = []
        train_contrast_acc = []
        for event in my_summary_iterator(summary_path):
            for value in event.summary.value:
                if value.tag == 'train_contrast_acc':
                    t = tensor_util.MakeNdarray(value.tensor)
                    steps.append(event.step)
                    train_contrast_acc.append(t)   
        if len(train_contrast_acc)>0:
            train_contrast_acc = np.stack(train_contrast_acc)
            steps = np.array(steps)
        else:
            train_contrast_acc =np.array([0])
            steps = np.array([0])
        return steps, train_contrast_acc
            
    def stop(self): 
        self._stop.set() 
    
    def stopped(self): 
        return self._stop.isSet() 

    def run(self): 
        # target function of the thread class 
        try:             
            the_process_already_started=False
            while not the_process_already_started and (not self.stopped()):
                # Load the existing model checkpoints.
                model_name = [l for l in os.listdir(flags.FLAGS.path) if 'Model_Training' in l][0]
                chekpointfiles = [l for l in os.listdir(flags.FLAGS.path+model_name) if ('events' in l) and ('.v2' in l)]
                # Check if the process has already started
                if len(chekpointfiles)==0:
                    continue
                checkpoint_file = sorted(chekpointfiles, key=lambda t: -os.stat(flags.FLAGS.path+model_name+'/'+t).st_mtime)[0]
                summary_path = flags.FLAGS.path+model_name+'/'+checkpoint_file
                
                # Check if training have already started.                   
                steps, train_contrast_acc = self.extract_train_contrast_acc(summary_path)
                last_step = steps[-1]
                train_contrast_acc_last = np.mean(train_contrast_acc[int(np.floor(train_contrast_acc.shape[0]*.9)):])
                # In case the CNN has started to be trained 
                if last_step>5:
                    the_process_already_started=True                    

            with tqdm(total=self.train_steps, ascii=True, desc="PCL: Train CNN (steps)") as bar_step:
                with tqdm(total= 1000,ascii=True, desc="PCL: Contrast Accuracy (per mille)") as bar_acc:
                    bar_step.update(last_step)
                    bar_acc.update(int(train_contrast_acc_last*1000))
                    
                    while bar_step.last_print_n<self.train_steps*.95 and (not self.stopped()):
                        # Extract train contrast accuracy and step information
                        model_name = [l for l in os.listdir(flags.FLAGS.path) if 'Model_Training' in l][0]
                        chekpointfiles = [l for l in os.listdir(flags.FLAGS.path+model_name) if ('events' in l) and ('.v2' in l)]
                        checkpoint_file = sorted(chekpointfiles, key=lambda t: -os.stat(flags.FLAGS.path+model_name+'/'+t).st_mtime)[0]                        
                        summary_path = flags.FLAGS.path+model_name+'/'+checkpoint_file
                        steps, train_contrast_acc = self.extract_train_contrast_acc(summary_path)
                        
                        # Update step
                        step_diff = steps[-1]-last_step
                        last_step = steps[-1]
                        bar_step.update(step_diff)

                        # Update contrast accuracy
                        train_contrast_acc_diff = train_contrast_acc_last - np.mean(train_contrast_acc[int(np.floor(train_contrast_acc.shape[0]*.9)):])
                        train_contrast_acc_last = np.mean(train_contrast_acc[int(np.floor(train_contrast_acc.shape[0]*.9)):])
                        if int(train_contrast_acc_diff*1000)<0:
                            train_contrast_acc_diff=0
                        bar_acc.update(int(train_contrast_acc_diff*1000))

                        time.sleep(5)
        finally: 
            pass

def PCL_execute(argv): 
    '''
    Train a CNN until the stops improving.
    '''
    # Intialize the dataset
    dataset = MyCustomDataset(flags.FLAGS.path, flags.FLAGS.n_images, flags.FLAGS.patch_size)

    # Obtain number of patches in the cohort
    total_num_patches=0
    for n_p_inI in dataset.num_patches_inImage:
        total_num_patches += dataset.num_patches_inImage[n_p_inI]

    # Number of training patches per epoch
    num_patches_epoch = int(flags.FLAGS.n_images*(total_num_patches/len(dataset.num_patches_inImage))*0.01)
    
    # Number of steps necessary to train/infer the model
    train_steps = model_util.get_train_steps(num_patches_epoch)    
    epoch_steps = int(round(num_patches_epoch / flags.FLAGS.train_batch_size))    

    # Check if it is first time training
    if len([l for l in os.listdir(flags.FLAGS.path) if 'Model_Training' in l])==0: # In case there is no model folder
        create_new_model = True
    elif len([nm for nm in os.listdir(flags.FLAGS.path+[l for l in os.listdir(flags.FLAGS.path) if 'Model_Training' in l][0]) if 'model' in nm])==0: # In case the model folder is empty
        create_new_model = True
    else:
        create_new_model = False

    # Create a new model
    if create_new_model:    
        # Add random letters to new directory
        letters = string.ascii_lowercase
        tmp = 'Model_Training_'+''.join(random.choice(letters) for i in range(4))+'/'
        folder_model_dir = flags.FLAGS.path+tmp
        
        # Create directory
        os.mkdir(folder_model_dir)
        progress = progress_bar(train_steps) 
        progress.start() 
        estimator = load_PCL_model(epoch_steps,folder_model_dir,num_patches_epoch,dataset,None)
        estimator.train(data_lib.load_patches_for_step(True, flags.FLAGS.train_batch_size, dataset, flags.FLAGS.patch_size,flags.FLAGS.n_images_iteration),max_steps=train_steps)        
        progress.stop()            
    else:
        # Obtain name of the model
        folder_model_dir = flags.FLAGS.path + [l for l in os.listdir(flags.FLAGS.path) if 'Model_Training' in l][0]

    while True:
        train_or_infer, checkpoint_dir = training_or_inference(train_steps)
        if train_or_infer:
            progress = progress_bar(train_steps) 
            progress.start() 
            estimator = load_PCL_model(epoch_steps,folder_model_dir,num_patches_epoch,dataset,checkpoint_dir)
            estimator.train(data_lib.load_patches_for_step(True, flags.FLAGS.train_batch_size, dataset, flags.FLAGS.patch_size,flags.FLAGS.n_images_iteration),max_steps=train_steps)        
            progress.stop()            
            
        else:                
            # Evaluate the model using the last file.
            estimator = load_PCL_model(epoch_steps,folder_model_dir,num_patches_epoch,dataset,checkpoint_dir)
            perform_evaluation_v4(estimator=estimator, input_fn=data_lib.build_input_fn_CHURRO_eval_nfile, dataset=dataset)
            break
    
def patch_contrastive_learning(path,params):
    '''
    Patch contrastive learning generates patch embeddings
    path: (string) that specifies the path of the folder 
    params: (list of strings) that specifies parameters of the file
    '''           
    
    # Obtain number of images that were preprocessed
    n_images = len([p for p in os.listdir(path+'Patch_Contrastive_Learning/Preprocessed_Images/') if '.npy' in p])

    flags.FLAGS.unparse_flags()
    # Define Flags.
    define_flags(path,params,n_images)
    
    # Disable eager mode when running with TF2.
    tf.disable_eager_execution()    

    # Run patch_contrastive_learning    
    app.run(PCL_execute)    
