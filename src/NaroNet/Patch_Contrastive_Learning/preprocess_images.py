"""
This module normalizes the cohort based on the mean and standard deviation.
"""

import os
import numpy as np
import random
from tqdm import tqdm
import csv
from NaroNet.utils.parallel_process import parallel_process
import NaroNet.utils.utilz as utilz
from skimage import io
import time

def loadImage(path,Channels):
    '''
    path: (string) that specifies the image path
    Channels: (vector of int) channels that should be included in the experiment.    
    '''

    # Load Image in its own format.
    if path.split('.')[-1]=='tiff':
        image = io.imread(path)    
    elif path.split('.')[-1]=='tif':
        image = io.imread(path)    
    elif path.split('.')[-1]=='npy':
        image = np.load(path)

    if len(image.shape)==3:
        # The 3rd dimension should be the channel dimension
        if np.argmin(image.shape)==0:
            shp = image.shape
            image = np.reshape(image,(image.shape[1]*image.shape[2],image.shape[0]))
            image = np.reshape(image,(shp[1],shp[2],shp[0]))
        elif np.argmin(image.shape)==1:
            image = np.reshape(image,(image.shape[0]*image.shape[2],image.shape[1]))
    
    elif len(image.shape)==4:
        shp = image.shape
        image = image.reshape((shp[1],shp[0],shp[2],shp[3]))
    
    # The histogram calculation needs positive values. SRY
    image = image-image.min(tuple(range(1,len(image.shape))),keepdims=True)

    # Eliminate unwanted channels
    if len(image.shape)==3:
        return image[:,:,Channels]
    if len(image.shape)==4:
        return image[Channels,:,:,:]

def Mean_std_experiment(base_path,image_paths,Channels):    
    ''' 
    Obtain mean and standard deviation from the cohort
    base_path: (string) that specifies the directory where the experiment is carried out.
    images_paths: (list of strings) that specifies the names of the files executed.
    Channels: (vector of int) channels that should be included in the experiment.    
    '''
    
    # Read slide by slide
    for n_im in tqdm(range(len(image_paths)),ascii=True,desc='Calculate Mean and Standard deviation'): 
        
        # Load Image
        image = loadImage(base_path+image_paths[n_im],Channels)        
                
        # To concatenate image information we sum the histograms of several images.
        if n_im==0:
            minImage = image.min(tuple(range(len(image.shape)-1)))
            minImage = [m*10 if m<0 else m/10 for m in minImage]
            maxImage = image.max(tuple(range(len(image.shape)-1)))
            maxImage = [m/10 if m<0 else m*10 for m in maxImage]
            Global_hist = [list(np.histogram(np.concatenate((image[:,:,i].flatten(),np.arange(minImage[i],maxImage[i],(maxImage[i]-minImage[i])/1000000))),range=(minImage[i],maxImage[i]),bins=1000000)) for i in range(image.shape[-1])]                                    
        else:
            Local_hist = [list(np.histogram(np.concatenate((image[:,:,i].flatten(),np.arange(minImage[i],maxImage[i],(maxImage[i]-minImage[i])/1000000))),range=(minImage[i],maxImage[i]),bins=1000000)) for i in range(image.shape[-1])]                                    
            for n_g_h, g_h in enumerate(Global_hist):
                g_h[0] += Local_hist[n_g_h][0]        

    # Calculate Mean
    mean = []    
    for g_h in Global_hist:
        hist_WA = []
        den = 0
        num = 0
        for g_n, g_h_h in enumerate(g_h[0]):
            den+=(g_h_h-1)
            num+=g_h[1][g_n]*(g_h_h-1)
        mean.append(num/den)
    
    # Calculate Standard deviation
    std = []
    for hn, g_h in enumerate(Global_hist):
        hist_WA = []
        den = 0
        num = 0
        for g_n, g_h_h in enumerate(g_h[0]):
            den+=(g_h_h-1)
            num+=((g_h[1][g_n]-mean[hn])**2)*(g_h_h-1)
        std.append((num/den)**0.5)

    return np.array(mean), np.array(std)

def apply_(n_im,base_path,image_paths,Channels,mean,std,output_path,patch_size,Z_score):
    # Load Image
    im = loadImage(base_path+image_paths[n_im],Channels)        
    
    # Apply Z-score normalization
    if len(im.shape)==3 and Z_score:
        x,y,chan = im.shape[0],im.shape[1],im.shape[2] 
        im = np.reshape(im,(x*y,chan))
        im = (im-mean)/(std+1e-16)
        im = np.reshape(im,(x,y,chan))
    elif len(im.shape)==4 and Z_score:
        im = (im - np.expand_dims(np.expand_dims(np.expand_dims(mean,axis=0),axis=0),axis=0))/(np.expand_dims(np.expand_dims(np.expand_dims(std,axis=0),axis=0),axis=0)+1e-16)        

    # Save Image
    np.save(output_path+'.'.join(image_paths[n_im].split('.')[:-1])+'.npy',im)

    # Assign number of patches per image.
    return n_im, int(im.shape[0]/patch_size)*int(im.shape[1]/patch_size)

def apply_zscoreNorm(base_path,output_path,image_paths,Channels,mean,std,patch_size,z_score):        
    '''
    As the title says, apply the z-score normalization to each image, so that the global mean of each marker is 0 and the std is 1. Save the image also.
    base_path: (string) that specifies the directory where the experiment is carried out.
    output_path: (string) that specifies the directory where the images are saved.
    images_paths: (list of strings) that specifies the names of the files executed.
    Channels: (vector of int) channels that should be included in the experiment.    
    mean: (array of int) that is the mean for each marker
    std: (array of int) that is the standard deviation for each marker
    patch_size: (int) that specifies the size of the patch
    '''

    # Obtain dicts of the parallel execution
    dict_zscore = []
    [dict_zscore.append({'n_im':i,'base_path':base_path,'image_paths':image_paths,'Channels':Channels,'mean':mean,'std':std,'output_path':output_path,'patch_size':patch_size,'Z_score':z_score}) for i in range(len(image_paths))]        

    num_patches_perImage = parallel_process(dict_zscore,apply_,use_kwargs=True,n_jobs=6,front_num=0,desc='Apply Z-score normalization')
    
    num_patches_perImage_p={}
    for n_im, n_patches in num_patches_perImage:
        # Assign number of patches per image.
        num_patches_perImage_p['.'.join(image_paths[n_im].split('.')[:-1])+'.npy']=n_patches
    
    return num_patches_perImage_p

def preprocess_images(path,ZScoreNormalization,patch_size):
    '''
    path: (path) where is the experiment to execute
    ZScoreNormalization: (boolean) Whether to normalize the images or not.
    Images_Names_Ends_In: (string) that specifies the image type    
    patch_size: (int) that specifies the size of the patch
    '''

    # Paths                            
    base_path = path+'Raw_Data/'
    output_path = path+'Patch_Contrastive_Learning/Preprocessed_Images/'
    
    # Create dir
    if not os.path.exists(path+'Patch_Contrastive_Learning'):
        os.mkdir(path+'Patch_Contrastive_Learning')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Obtain Image Paths
    image_paths = os.listdir(base_path+'Images')    
    preprocessed_paths = [i for i in os.listdir(output_path) if '.npy' in i]
    z_score = True # Apply z-score normalization or not

    Channels, Marker_Names = utilz.load_channels(base_path)
    
    #Preprocess images
    if len(image_paths)!=len(preprocessed_paths):

        # Iterate Images to obtain mean and std of marker distribution.
        random.shuffle(image_paths)
        print('Preprocess a cohort of ',str(len(image_paths)),' subjects:')
        mean, std = Mean_std_experiment(base_path=base_path+'Images/',image_paths=image_paths,Channels=Channels)

        # Apply z-score normalization and save them to efficient structures.
        num_patches_perImage = apply_zscoreNorm(base_path+'Images/',output_path,image_paths,Channels,mean,std,patch_size,z_score)        
        
        # Write num_patches_perImage to csv.
        w = csv.writer(open(output_path+"Num_patches_perImage.csv", "w"))
        for key, val in num_patches_perImage.items():
            w.writerow([key, val])
        time.sleep(5)
            
    else:        
        with tqdm(total=len(image_paths), ascii=True, desc='Calculate Mean and Standard deviation') as bar_folds:            
            bar_folds.update(len(image_paths))
        with tqdm(total=len(image_paths), ascii=True, desc='Apply Z-score normalization') as bar_folds:            
            bar_folds.update(len(image_paths))
