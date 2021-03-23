import numpy as np
import os.path as osp
import copy
import os
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm
from scipy import stats
import seaborn as sns
from NaroNet.utils.parallel_process import parallel_process
from tifffile.tifffile import imwrite
import pandas as pd
import itertools
from skimage import filters

def load_cell_types_assignments(dataset, cell_type_idx, subject_info,subgraph_idx,n_cell_types, prev_cell_type_assignment):
    """
    Obtain matrix that assigns patches to cell types (phenotypes and neighborhoods)
    dataset: (object)
    cell_type_idx: (int) cell_type_idx==0 is phenotype, cell_type_idx==1 is neihborhood, cell_type_idx==2 is neihborhood interaction
    subject_info: (list of str and int)
    subgraph_idx: (int) specifying the index of the subgraph.
    n_cell_types: (int) number of cell types (phenotypes or neighborhoods)
    prev_cell_type_assignment: (array of int) specifying assignments
    """

    # If phenotype or neighborhood load matrix assignment.
    if cell_type_idx<2:
        cell_type_assignment = np.load(osp.join(dataset.processed_dir_cell_types,
                    'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(subject_info[1], subgraph_idx, n_cell_types)))                        

    # if neihborhood interaction the matrix assigns neighbors to neighborhood interactions.
    else:        
        # Load neighborhood interactions
        secondorder_assignment = np.load(osp.join(dataset.processed_dir_cell_types,
                                'cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(subject_info[1], n_cell_types)))                    
        
        # Obtain assignment of patches to neighborhood interactions
        cell_type_assignment = np.matmul(prev_cell_type_assignment,secondorder_assignment)
    prev_cell_type_assignment = copy.deepcopy(cell_type_assignment)
    
    return prev_cell_type_assignment, cell_type_assignment

def load_patch_image(dataset,subject_info,im,imList):
    '''
    Load patch image
    '''

    if ('Synthetic' in dataset.root) or ('Endometrial_Low' in dataset.root):                                            
        if len(imList)==0:
            suprpxlIm = np.zeros(im.shape[:2])                        
            # Create superPatch Label image using the Superpatch size. 
            division = np.floor(suprpxlIm.shape[0]/dataset.patch_size)
            lins = np.repeat(list(range(int(division))), dataset.patch_size)
            lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
            for y_indx, y in enumerate(range(int(division))):
                suprpxlIm[:int(division*dataset.patch_size),y_indx*dataset.patch_size:(y_indx+1)*dataset.patch_size] = lins+int(y_indx*division)
            if ('V2' in dataset.root) or ('Endometrial_Low' in dataset.root) or ('V4' in dataset.root):
                suprpxlIm = np.transpose(suprpxlIm)
            suprpxlIm = suprpxlIm.astype(int)
            numberOfMarkers = np.ones(im.shape[2])
        else:
            for imList_n ,imList_i in enumerate(imList):
                suprpxlIm_i = np.zeros(imList_i.shape[:2])                        
                # Create superPatch Label image using the Superpatch size. 
                division = np.floor(suprpxlIm_i.shape[0]/dataset.patch_size)
                lins = np.repeat(list(range(int(division))), dataset.patch_size)
                lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
                for y_indx, y in enumerate(range(int(division))):
                    suprpxlIm_i[:int(division*dataset.patch_size),y_indx*dataset.patch_size:(y_indx+1)*dataset.patch_size] = lins+int(y_indx*division)
                if (not 'V2' in dataset.root) or (not 'V4' in dataset.root):
                    suprpxlIm_i = np.transpose(suprpxlIm_i)
                suprpxlIm_i = suprpxlIm_i.astype(int) 
                if imList_n==0:
                    suprpxlIm = suprpxlIm_i
                else:
                    suprpxlIm = np.concatenate((suprpxlIm, suprpxlIm_i+suprpxlIm.max()+1),1)
            # suprpxlIm=suprpxlIm+1
            numberOfMarkers = np.ones(im.shape[2])
    elif ('Endometrial_POLE' in dataset.root) or ('Lung' in dataset.root):
        suprpxlIm = np.zeros(im.shape[:2])                        
        # Create superPatch Label image using the Superpatch size. 
        # Create superPatch Label image using the Superpatch size. 
        division_rows = np.floor(suprpxlIm.shape[0]/dataset.patch_size)
        division_cols = np.floor(suprpxlIm.shape[1]/dataset.patch_size)
        lins = np.repeat(list(range(int(division_cols))), dataset.patch_size)
        lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
        for row_indx, row in enumerate(range(int(division_rows))):
            suprpxlIm[row_indx*dataset.patch_size:(row_indx+1)*dataset.patch_size,:int(division_cols*dataset.patch_size)] = np.transpose(lins+int(row_indx*division_cols))
        # suprpxlIm = np.transpose(suprpxlIm)
        suprpxlIm = suprpxlIm.astype(int)             
        numberOfMarkers = np.ones(im.shape[2])                       

    elif 'ZuriBasel' in dataset.root or True:
        # if 'Superpixel' in dataset.raw_dir:
        #     suprpxlIm = np.load(osp.join(dataset.root,'OriginalSuperpixel','{}.npy'.format('Labels_'+subject_info[0][11:])))
        # elif 'SuperPatch' in dataset.raw_dir:
        # Create Mosaic                            
        x_dim = max([i.shape[0] for i in imList])
        y_dim = sum([i.shape[1] for i in imList])
        suprpxlIm = np.zeros((x_dim,y_dim))
        last_y = 0
        
        for imList_n ,imList_i in enumerate(imList):
            suprpxlIm_i = np.zeros(imList_i.shape[:2])                        
            # Create superPatch Label image using the Superpatch size. 
            division = np.floor(suprpxlIm_i.shape[0]/dataset.patch_size)
            lins = np.repeat(list(range(int(division))), dataset.patch_size)
            lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
            for y_indx, y in enumerate(range(int(np.floor(suprpxlIm_i.shape[1]/dataset.patch_size)))):
                suprpxlIm_i[:int(division*dataset.patch_size),y_indx*dataset.patch_size:(y_indx+1)*dataset.patch_size] = lins+int(y_indx*division)
            # suprpxlIm_i = np.transpose(suprpxlIm_i)
            suprpxlIm_i = suprpxlIm_i.astype(int) 
            if len(imList)==0:
                suprpxlIm = suprpxlIm_i
            else:                                                                   
                suprpxlIm[:suprpxlIm_i.shape[0],last_y:last_y+suprpxlIm_i.shape[1]] = suprpxlIm_i[:,:]
                last_y = last_y + suprpxlIm_i.shape[1]           
        suprpxlIm=suprpxlIm+1
        suprpxlIm = suprpxlIm.astype(int)
        numberOfMarkers = np.ones(im.shape[2])   

    return suprpxlIm,numberOfMarkers

def select_patches_from_cohort_(clusters,dataset,subject_info,count,CropConfPheno,CropConfTissueComm):

    # Crops,Confidence,and Phenotype Vector.    
    prev_cell_type_assignment=[]
    
    for cell_type_idx, n_cell_types in enumerate(clusters[:2]):    
        # Apply mask to patch
        for subgraph_idx in range(dataset.findLastIndex(subject_info[1])+1):                
            # Open Raw Image.
            im, imList = dataset.open_Raw_Image(subject_info,1)                
            # load cell_types_assignments
            prev_cell_type_assignment, cell_type_assignment = load_cell_types_assignments(
                                                                    dataset,cell_type_idx,subject_info,subgraph_idx,
                                                                    n_cell_types,prev_cell_type_assignment)                
            # Load image of patches
            Patch_im,Markers = load_patch_image(dataset,subject_info,im,imList)                    
            # Open Single-Cell Contrastive Learning Information
            PCL_reprsntions = np.load(dataset.raw_dir+'/{}.npy'.format(subject_info[0]))                     
            # Select Patches of a specific Phenotype and save its confidence
            cell_type_top1 = np.apply_along_axis(np.argmax, axis=1, arr=cell_type_assignment) 
            # Select the the patches with most confidence for each 
            CropConfPheno,CropConfTissueComm = topk_confident_patches(dataset,n_cell_types,cell_type_top1,Patch_im,
                                                                        cell_type_assignment,cell_type_idx,CropConfPheno,
                                                                        CropConfTissueComm,im,PCL_reprsntions,count)  
    return CropConfPheno,CropConfTissueComm


def select_patches_from_cohort(dataset,IndexAndClass,clusters):
    '''
    '''
    # Crops,Confidence,and Phenotype Vector.
    CropConfPheno = []
    CropConfTissueComm = []
    for c in range(clusters[0]):
        CropConfPheno.append([])
            
    for c in range(clusters[1]):
        CropConfTissueComm.append([])  

    # Prepare parallel process
    dict_subjects = []
    for count, subject_info in enumerate(IndexAndClass):
        if subject_info[2][0]!='None':
            dict_subjects.append({'clusters':clusters,'dataset':dataset,'subject_info':subject_info,'count':count,'CropConfPheno':CropConfPheno,'CropConfTissueComm':CropConfTissueComm})
    
    # select_patches_from_cohort
    result = parallel_process(dict_subjects,select_patches_from_cohort_,use_kwargs=True,front_num=0,desc='BioInsights: Get relevant examples of cell types') 

    # Get lists of patches
    for R in result:  
        for r_i, r in enumerate(R[0]):
            CropConfPheno[r_i].append(r)
        for r_i, r in enumerate(R[1]):
            CropConfTissueComm[r_i].append(r)
    
    # Join lists of patches
    for n_c, c in enumerate(CropConfPheno):
        aux_list = []
        for c_c in c:
            aux_list += c_c
        CropConfPheno[n_c] = copy.deepcopy(aux_list)
    for n_c, c in enumerate(CropConfTissueComm):
        aux_list = []
        for c_c in c:
            aux_list += c_c
        CropConfTissueComm[n_c] = copy.deepcopy(aux_list)
            
    return CropConfPheno,CropConfTissueComm

def topk_confident_patches(dataset,n_cell_types,cell_type_top1,Patch_im,
                            cell_type_assignment,cell_type_idx,CropConfPheno,
                            CropConfTissueComm,im,PCL_reprsntions,count):
    '''
    '''
    K = 10

    for c in range(n_cell_types):
                                                                        
        # Obtain patch info of the K with most certainty
        for patch_idx in np.where(cell_type_top1[:Patch_im.max()]==c)[0][cell_type_assignment[np.where(cell_type_top1[:Patch_im.max()]==c)[0],c].argsort()[-K:]]:                  
                    
            # Select the patch from the patch_image
            mask = Patch_im==patch_idx

            # Avoid selecting cropped images
            if mask.sum()!=dataset.patch_size**2:
                continue

            if cell_type_idx==0:
                CropConfPheno[c].append([im[mask.argmax(0).max():mask.argmax(0).max()+dataset.patch_size,mask.argmax(1).max():mask.argmax(1).max()+dataset.patch_size], # The original Image
                                            cell_type_assignment[patch_idx,c], # The cell type certainty
                                            PCL_reprsntions[patch_idx,:], # The parameters obtained from contrastive learning
                                            [100000*count+patch_idx]]) # Number of the image, and patch identificator
            elif cell_type_idx==1:
                minIdx = mask.argmax(0).max()-dataset.patch_size*2
                maxIdx = mask.argmax(0).max()+dataset.patch_size*3
                minIdy = mask.argmax(1).max()-dataset.patch_size*2
                maxIdy = mask.argmax(1).max()+dataset.patch_size*3
                if maxIdx>im.shape[0] or maxIdy>im.shape[1] or minIdx<0 or minIdy<0:
                    continue
                CropConfTissueComm[c].append([im[minIdx:maxIdx,minIdy:maxIdy], # The original Image
                                            cell_type_assignment[patch_idx,c], # The cell type certainty
                                            [100000*count+patch_idx]]) # Patch identificator 
    
    return CropConfPheno,CropConfTissueComm

def save_2Dmatrix_in_excel_with_names(filename,matrix,Names):
    dict_ = {}
    for n, name in enumerate(Names):
        dict_[name] = matrix[:,n]
    dict_ = pd.DataFrame.from_dict(dict_)      
    dict_.to_excel(filename) 

def save_heatmap_with_names(filename,matrix,Names):    
    if matrix.shape[0]>2:
        plt.close()
        plt.figure()
        sns.set(font_scale=1.1)
        h_E_Fig = sns.clustermap(matrix, col_cluster=True, row_cluster=True, xticklabels=Names, linewidths=0,vmin=-2, vmax=2, cmap="bwr")                
        h_E_Fig.savefig(filename,dpi=600) 

def calculate_IoU_usingOtsu(matrix_0,matrix_1):
    # Check if all values are equal and return an IoU of zeros
    if matrix_0.max()==matrix_0.min() or matrix_1.max()==matrix_1.min():
        return 0

    matrix_0 = matrix_0>filters.threshold_otsu(matrix_0)
    matrix_1 = matrix_1>filters.threshold_otsu(matrix_1)        
    intersection = np.logical_and(matrix_0, matrix_1)
    union = np.logical_or(matrix_0, matrix_1)
    return np.sum(intersection) / np.sum(union)    

def calculate_marker_colocalization(filename,matrix,MarkerNames):
    # matrix: contains (number of patches, x_dimension, y_dimension, number of markers)
    Marker_Colocalization =  np.zeros(len(list(itertools.combinations(MarkerNames,2))))
    for n_comb, pair_of_markers in enumerate(itertools.combinations(MarkerNames,2)):
        id_0 = MarkerNames.index(pair_of_markers[0])
        id_1 = MarkerNames.index(pair_of_markers[1])
        matrix_0 = matrix[:,:,:,[id_0]]
        matrix_1 = matrix[:,:,:,[id_1]]
        for n_patch in range(matrix.shape[0]):            
            Marker_Colocalization[n_comb] += calculate_IoU_usingOtsu(matrix_0[n_patch,:,:],matrix_1[n_patch,:,:])
    return Marker_Colocalization/matrix.shape[0]

def extract_topk_patches_from_cohort(dataset, CropConf, Marker_Names,cell_type):
    '''
    docstring
    '''

    thisfolder = dataset.bioInsights_dir_cell_types + cell_type+'/'

    if cell_type=='Phenotypes':
        mult_1 = 1 
        mult_2 = 2
    else:
        mult_1 = 5 
        mult_2 = 4

    ## Iterate through Phenotypes to extract topk patches
    k=400
    topkPatches=[]
    # Create a heatmap marker using topk patches
    heatmapMarkerExpression = np.zeros((7,len(CropConf),len(Marker_Names))) #Number of TMEs x number of markers
    heatmap_Colocalization = np.zeros((len(CropConf),int(len(Marker_Names)*(len(Marker_Names)-1)/2))) # Number of TMEs x Number of Marker combinations

    # Mean marker expression and std marker expression
    AllCELLLS = np.concatenate([np.stack([c[0].mean((0,1)) for c in CC]) for CC in CropConf if len(CC)>0])

    # Use CropCOnf, that saves a lot of patches...
    for n_cell_type ,CropConf_i in enumerate(CropConf):
        
        if len(CropConf_i)==0:
            continue
        
        # Choose patches with most confidence
        topkPheno = np.array([CCP[1] for CCP in CropConf_i]).argsort()[-k:]        
        # Save topkPheno to heatMarkerMap. Mean
        MarkerExpression = np.array([CropConf_i[t][0] for t in topkPheno])
        Confidence = np.array([CropConf_i[t][1] for t in topkPheno])

        heatmap_Colocalization[n_cell_type,:] = calculate_marker_colocalization(thisfolder+'TME_{}_Marker_Colocalization.xlsx'.format(n_cell_type+1),MarkerExpression,Marker_Names)

        # Save marker expression of patches individually.
        save_2Dmatrix_in_excel_with_names(thisfolder+'TME_{}_Conf_{}.xlsx'.format(n_cell_type+1,Confidence.mean().round(2)),MarkerExpression.mean((1,2)),Marker_Names)
        save_heatmap_with_names(thisfolder+'TME_{}_Conf_{}.png'.format(n_cell_type+1,Confidence.mean().round(2)),(MarkerExpression.mean((1,2))-AllCELLLS.mean(0))/AllCELLLS.std(0),Marker_Names)

        MarkerExpression = np.reshape(MarkerExpression,(MarkerExpression.shape[0]*MarkerExpression.shape[1]*MarkerExpression.shape[2],MarkerExpression.shape[3]))
        heatmapMarkerExpression[0,n_cell_type,:] = np.mean(MarkerExpression,axis=0)                    
        for n_i, i in enumerate([50,99]):
            heatmapMarkerExpression[n_i+1,n_cell_type,:] = np.percentile(MarkerExpression,i,axis=0)                    

        # Save Image in RGB
        ImwithKPatches = np.zeros((dataset.patch_size*mult_1*int(np.sqrt(k))+mult_2*int(np.sqrt(k)),dataset.patch_size*mult_1*int(np.sqrt(k))+mult_2*int(np.sqrt(k)),CropConf_i[0][0].shape[2]))
        ImwithKPatches_Norm_perPatch = np.zeros((dataset.patch_size*mult_1*int(np.sqrt(k))+mult_2*int(np.sqrt(k)),dataset.patch_size*mult_1*int(np.sqrt(k))+mult_2*int(np.sqrt(k)),CropConf_i[0][0].shape[2]))
        for t_n, t in enumerate(topkPheno):
            row = np.floor(t_n/int(k**0.5))
            col = np.mod(t_n,int(k**0.5))
       
            # Assign patch to Image
            ImwithKPatches[int(row*mult_1*dataset.patch_size+row*mult_2):int((row+1)*mult_1*dataset.patch_size+row*mult_2),int(col*mult_1*dataset.patch_size+col*mult_2):int((col+1)*mult_1*dataset.patch_size+col*mult_2),:] = CropConf_i[t][0]
            ImwithKPatches_Norm_perPatch[int(row*mult_1*dataset.patch_size+row*mult_2):int((row+1)*mult_1*dataset.patch_size+row*mult_2),int(col*mult_1*dataset.patch_size+col*mult_2):int((col+1)*mult_1*dataset.patch_size+col*mult_2),:] = CropConf_i[t][0]/CropConf_i[t][0].max((0,1),keepdims=True)
        
        if len(Marker_Names)<10:
            # Fill unassigned patches with zeroes.
            for t_n in range(len(topkPheno),k):
                row = np.floor(t_n/int(k**0.5))
                col = np.mod(t_n,int(k**0.5))
                # Assign patch to Image
                ImwithKPatches[int(row*mult_1*dataset.patch_size+row*mult_2):int((row+1)*mult_1*dataset.patch_size+row*mult_2),int(col*mult_1*dataset.patch_size+col*mult_2):int((col+1)*mult_1*dataset.patch_size+col*mult_2),:] = 0.0
                ImwithKPatches_Norm_perPatch[int(row*mult_1*dataset.patch_size+row*mult_2):int((row+1)*mult_1*dataset.patch_size+row*mult_2),int(col*mult_1*dataset.patch_size+col*mult_2):int((col+1)*mult_1*dataset.patch_size+col*mult_2),:] = 0.0
                            
            # RGBImwithKPatches = dataset.nPlex2RGB(ImwithKPatches)
            imwrite(thisfolder+'Cell_type_{}_Raw.tiff'.format(n_cell_type+1),np.moveaxis(ImwithKPatches,2,0))
            imwrite(thisfolder+'Cell_type_{}_Patch_Norm.tiff'.format(n_cell_type+1),np.moveaxis(ImwithKPatches_Norm_perPatch,2,0))
            
            # Save Certainty of this Phenotype
            plt.close()
            plt.figure()
            n, bins, patches = plt.hist(np.array([i[1] for i in CropConf_i]), 100, color=cm.jet_r(int(n_cell_type*(255/int(len(CropConf))))), alpha=1)            
            plt.ylabel('Number of Superpatches',fontsize=16)            
            plt.xlabel('Level of cell type certainty',fontsize=16)
            plt.title('Histogram of phenotype '+str(n_cell_type+1)+' Certainty',fontsize=16)
            plt.savefig(thisfolder+'ConfidenceHistogram_{}.png'.format(n_cell_type+1), format="PNG",dpi=600)                                         

        # Assign topkPheno Patches
        topkPatches+=[CropConf_i[t] for t in topkPheno]     

    return heatmapMarkerExpression, heatmap_Colocalization

def save_heatmap_raw_and_normalized(filename, heatmap, TME_names,Colormap,Marker_Names):    
    y_ticklabels = TME_names[heatmap.sum(1)!=0]
    c_map = Colormap[:heatmap.shape[0]][heatmap.sum(1)!=0]
    # Figure Heatmap Raw Values
    plt.close()
    plt.figure()
    sns.set(font_scale=1.1)
    h_E_Fig = sns.clustermap(heatmap[heatmap.sum(1)!=0,:],col_cluster=True, row_cluster=True, row_colors=c_map,xticklabels=Marker_Names,yticklabels=y_ticklabels, linewidths=0.5, cmap="Spectral_r")            
    h_E_Fig.savefig(filename+'_Raw.png',dpi=600) 

    # Figure Heatmap Min is 0 and max is 1 Values    
    h_E_COL_MinMax = heatmap[heatmap.sum(1)!=0,:] - heatmap[heatmap.sum(1)!=0,:].min(0,keepdims=True)
    h_E_COL_MinMax = h_E_COL_MinMax/h_E_COL_MinMax.max(0,keepdims=True)
    h_E_COL_MinMax[np.isnan(h_E_COL_MinMax)] = 0 
    plt.close()
    plt.figure()
    sns.set(font_scale=1.1)
    h_E_Fig = sns.clustermap(h_E_COL_MinMax[h_E_COL_MinMax.sum(1)!=0,:],col_cluster=True, row_cluster=True, row_colors=c_map,xticklabels=Marker_Names,yticklabels=y_ticklabels, linewidths=0.5, cmap="Spectral_r")            
    h_E_Fig.savefig(filename+'_MinMax.png',dpi=600) 
    
    # Figure Heatmap z-scored values
    h_E_COL_Norm = stats.zscore(heatmap[heatmap.sum(1)!=0,:],axis=0)  
    h_E_COL_Norm[np.isnan(h_E_COL_Norm)] = 0              
    plt.close()
    plt.figure()
    sns.set(font_scale=1.1)
    h_E_Fig = sns.clustermap(h_E_COL_Norm,col_cluster=True, vmin=-2, vmax=2, row_cluster=True, row_colors=c_map,xticklabels=Marker_Names,yticklabels=y_ticklabels, linewidths=0.5, cmap="Spectral_r")            
    h_E_Fig.savefig(filename+'_Norm.png',dpi=600) 

def save_heatMapMarker_and_barplot(dataset, heatmapMarkerExpression, heatmapMarkerColocalization,CropConf,Marker_Names,cell_type):
    '''
    '''
    if cell_type=='Phenotypes':
        abrev = 'P'
    else:
        abrev = 'N'

    Colormap=cm.jet(range(0,255,int(255/heatmapMarkerExpression.shape[1])))[:,:3]

    # Save heatmapmarker to disk
    for n_j, j in enumerate([25,50,99]):
        save_heatmap_raw_and_normalized(dataset.bioInsights_dir_cell_types+cell_type+'/heatmap_MarkerExpression_'+str(j), 
                                        heatmapMarkerExpression[n_j,:,:], np.array([abrev+str(i+1) for i in range(len(CropConf))]),
                                        Colormap,Marker_Names)
        save_heatmap_raw_and_normalized(dataset.bioInsights_dir_cell_types+cell_type+'/heatmap_MarkerExpression_Colocalization_'+str(j), 
                                        np.concatenate((heatmapMarkerExpression[n_j,:,:],heatmapMarkerColocalization),axis=1), np.array([abrev+str(i+1) for i in range(len(CropConf))]),
                                        Colormap,Marker_Names+['_'.join(i) for i in itertools.combinations(Marker_Names,2)])
    
    save_heatmap_raw_and_normalized(dataset.bioInsights_dir_cell_types + cell_type + '/heatmap_Colocalization', heatmapMarkerColocalization, np.array([abrev+str(i+1) for i in range(len(CropConf))]),Colormap,['_'.join(i) for i in itertools.combinations(Marker_Names,2)])
        # h_E = heatmapMarkerExpression[n_j,:,:]                 
        # y_ticklabels = np.array([abrev+str(i+1) for i in range(len(CropConf))])[h_E.sum(1)!=0]
        # c_map = Colormap[:h_E.shape[0]][h_E.sum(1)!=0]
        # plt.close()
        # plt.figure()
        # sns.set(font_scale=1.1)
        # h_E_Fig = sns.clustermap(h_E[h_E.sum(1)!=0,:],col_cluster=True, row_cluster=True, row_colors=c_map,xticklabels=Marker_Names,yticklabels=y_ticklabels, linewidths=0.5, cmap="bwr")            
        # h_E_Fig.savefig(dataset.bioInsights_dir_cell_types + cell_type + '/heatmap_MarkerExpression_{}_Raw.png'.format(str(j)),dpi=600) 
        # h_E_COL_Norm = stats.zscore(h_E[h_E.sum(1)!=0,:],axis=0)  
        # h_E_COL_Norm[np.isnan(h_E_COL_Norm)] = 0              
        # plt.close()
        # plt.figure()
        # sns.set(font_scale=1.1)
        # h_E_Fig = sns.clustermap(h_E_COL_Norm,col_cluster=True, vmin=-2, vmax=2, row_cluster=True, row_colors=c_map,xticklabels=Marker_Names,yticklabels=y_ticklabels, linewidths=0.5, cmap="bwr")            
        # h_E_Fig.savefig(dataset.bioInsights_dir_cell_types + cell_type + '/heatmap_MarkerExpression_{}_Col_Norm.png'.format(str(j)),dpi=600) 

    # Calculate Phenotype abundance across all patients
    plt.close()
    plt.figure()
    sns.set(font_scale=1.0)
    BarPlotPresenceOfPhenotypes = sns.barplot(x=[abrev+str(i+1) for i in range(len(CropConf))] ,y=np.array([len(i) for i in CropConf]),palette=Colormap)      
    BarPlotPresenceOfPhenotypes.set(xlabel=cell_type, ylabel = 'Number of patches',title='Histogram of abundance across the patient cohort')                
    BarPlotPresenceOfPhenotypes.set_xticklabels([abrev+str(i+1) for i in range(len(CropConf))], size=7)
    plt.savefig(dataset.bioInsights_dir_cell_types + cell_type +'/Barplot_cell_types.png',dpi=600)

def neigh_comp(TC,phenoInd):
    InteractivityVect = np.zeros(len(phenoInd))
    for n_Phen, PH in enumerate(phenoInd):
        PH = [p[0] for p in PH]
        if len(PH)>0:
            for t in TC:
                if t[2] in PH:
                    InteractivityVect[n_Phen]+=1     
    return InteractivityVect

def obtain_neighborhood_composition(dataset,CropConfPheno,CropConfTissueComm):
    '''
    '''

    # Find indices of phenotypes
    phenoInd = []
    for c in CropConfPheno:
        # Find all indices
        phenoInd.append([c_n[3] for c_n in c])        

    # Generate Interactivity matrix
    dict_neigh = []
    for n_Neighbor, TC in enumerate(CropConfTissueComm):
        dict_neigh.append({'TC':TC,'phenoInd':phenoInd})
    result = parallel_process(dict_neigh,neigh_comp,use_kwargs=True,front_num=0,desc='BioInsights: Calculate phenotype abundance whithin neighborhoods') 
    InteractivityMatrix = np.stack(result)

    # Input
    Colormap_Pheno=cm.jet_r(range(0,255,int(255/len(CropConfPheno))))[:,:3]
    Colormap_Neigh=cm.jet_r(range(0,255,int(255/len(CropConfTissueComm))))[:,:3]

    # Save interactivity matrix
    sns.set(font_scale=1.5)
    plt.close()
    heatmapInteractivityMatrix_Fig = sns.clustermap(InteractivityMatrix,col_cluster=False, row_cluster=False, row_colors=Colormap_Neigh, col_colors=Colormap_Pheno,xticklabels=['P'+str(i+1) for i in range(len(CropConfPheno))],yticklabels=['N'+str(i+1) for i in range(len(CropConfTissueComm))], linewidths=0.5, cmap="bwr")            
    plt.xlabel("Phenotypes")
    plt.ylabel("Neighborhoods")
    heatmapInteractivityMatrix_Fig.savefig(dataset.bioInsights_dir_cell_types+'Neighborhoods/heatmap_InteractivityMat_Raw.png',dpi=600) 
    plt.close()
    InteractivityMatrix[InteractivityMatrix==0]=1e-3
    heatmapInteractivityMatrix_Fig = sns.clustermap(stats.zscore(InteractivityMatrix,axis=1),col_cluster=False, row_cluster=False, row_colors=Colormap_Neigh, col_colors=Colormap_Pheno,xticklabels=['P'+str(i+1) for i in range(len(CropConfPheno))],yticklabels=['N'+str(i+1) for i in range(len(CropConfTissueComm))], linewidths=0.5, cmap="bwr")            
    plt.xlabel("Phenotypes")
    plt.ylabel("Neighborhoods")
    heatmapInteractivityMatrix_Fig.savefig(dataset.bioInsights_dir_cell_types+'Neighborhoods/heatmap_InteractivityMat_Norm.png',dpi=600) 

def Area_to_Neighborhood_to_Phenotype(dataset,clusters,IndexAndClass,num_classes, ClusteringThreshold):
    # Patient-Type            
    Area_to_Neigh = np.zeros((clusters[-1],clusters[-2]))                
    Neigh_to_Pheno = np.zeros((clusters[-2],clusters[-3]))            

    # Obtain clusters per Slide
    for count, idxclster in enumerate(IndexAndClass):           
        try:
            neigh_to_area_assignment = np.load(osp.join(dataset.processed_dir_cell_types,'cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(idxclster[1], clusters[-1])))                
            patch_to_neigh_assignment = np.load(osp.join(dataset.processed_dir_cell_types,'cluster_assignmentPerPatch_Index_{}_0_ClustLvl_{}.npy'.format(idxclster[1], clusters[-2])))        
            patch_to_pheno_assignment = np.load(osp.join(dataset.processed_dir_cell_types,'cluster_assignmentPerPatch_Index_{}_0_ClustLvl_{}.npy'.format(idxclster[1], clusters[-3])))                
        except:
            continue
        # Assign Neigh_to_Area
        neigh_to_area_assignment = neigh_to_area_assignment - neigh_to_area_assignment.min(-2)
        neigh_to_area_assignment = neigh_to_area_assignment / (neigh_to_area_assignment.max(-2,keepdims=True)+1e-12)        
        Area_to_Neigh += np.transpose(neigh_to_area_assignment)
        # Assign Patch_to_Neigh
        PercTHrsl_neigh = np.percentile(patch_to_neigh_assignment,axis=0,q=ClusteringThreshold)
        PercTHrsl_pheno = np.percentile(patch_to_pheno_assignment,axis=0,q=ClusteringThreshold)
        for i in range(patch_to_neigh_assignment.shape[1]):
            patch_to_neigh_assignment[:,i][PercTHrsl_neigh[i]>=patch_to_neigh_assignment[:,i]] = 0 
        for i in range(patch_to_pheno_assignment.shape[1]):
            patch_to_pheno_assignment[:,i][PercTHrsl_pheno[i]>=patch_to_pheno_assignment[:,i]] = 0         
        Neigh_to_Pheno += np.matmul(np.transpose(patch_to_neigh_assignment),patch_to_pheno_assignment)

    # Save heatmap Neigh_to_Pheno
    plt.close()
    plt.figure()
    sns.set(font_scale=1.1)
    row_colors_Neigh=cm.jet_r(range(0,255,int(255/clusters[1])))[:,:3]
    yticklabels_Neigh = ['N'+str(i+1) for i in range(clusters[1])]
    col_colors_Pheno=cm.jet_r(range(0,255,int(255/clusters[0])))[:,:3]
    xticklabels_Pheno = ['P'+str(i+1) for i in range(clusters[0])]
    Neigh_to_Pheno = Neigh_to_Pheno/(Neigh_to_Pheno.sum(1,keepdims=True)+1e-16)
    h_E_Fig = sns.clustermap(Neigh_to_Pheno.transpose(),col_cluster=True, row_cluster=True, col_colors=row_colors_Neigh,
                                row_colors=col_colors_Pheno,yticklabels=xticklabels_Pheno,xticklabels=yticklabels_Neigh, linewidths=0.5, cmap="bwr")            
    h_E_Fig.savefig(dataset.bioInsights_dir_cell_types + 'Neighborhoods/heatmap_Phenotype_composition_of_neighborhoods_Raw_Thrs{}.png'.format(str(ClusteringThreshold)),dpi=600) 

    # Save heatmap Neigh_to_Pheno Normalized
    plt.close()
    plt.figure()
    sns.set(font_scale=1.1)
    row_colors_Neigh=cm.jet_r(range(0,255,int(255/clusters[1])))[:,:3]
    yticklabels_Neigh = ['N'+str(i+1) for i in range(clusters[1])]
    col_colors_Pheno=cm.jet_r(range(0,255,int(255/clusters[0])))[:,:3]
    xticklabels_Pheno = ['P'+str(i+1) for i in range(clusters[0])]
    Neigh_to_Pheno = stats.zscore(Neigh_to_Pheno,axis=0)
    Neigh_to_Pheno[np.isnan(Neigh_to_Pheno)] = 0
    h_E_Fig = sns.clustermap(Neigh_to_Pheno.transpose(),col_cluster=True, row_cluster=True, col_colors=row_colors_Neigh,
                                row_colors=col_colors_Pheno,yticklabels=xticklabels_Pheno,xticklabels=yticklabels_Neigh, linewidths=0.5, cmap="bwr")            
    h_E_Fig.savefig(dataset.bioInsights_dir_cell_types + 'Neighborhoods/heatmap_Phenotype_composition_of_neighborhoods_Norm_Thrs{}.png'.format(str(ClusteringThreshold)),dpi=600) 

    # Save heatmap Area_to_Neigh
    plt.close()
    plt.figure()
    sns.set(font_scale=1.1)
    row_colors_Area=cm.jet_r(range(0,255,int(255/clusters[2])))[:,:3]
    yticklabels_Area = ['A'+str(i+1) for i in range(clusters[2])]
    col_colors_Neigh=cm.jet_r(range(0,255,int(255/clusters[1])))[:,:3]
    xticklabels_Neigh = ['N'+str(i+1) for i in range(clusters[1])]
    Area_to_Neigh = Area_to_Neigh/(Area_to_Neigh.sum(1,keepdims=True)+1e-16)
    h_E_Fig = sns.clustermap(Area_to_Neigh.transpose(),col_cluster=True, row_cluster=True, col_colors=row_colors_Area,
                                row_colors=col_colors_Neigh,yticklabels=xticklabels_Neigh,xticklabels=yticklabels_Area, linewidths=0.5, cmap="bwr")            
    if not os.path.exists(dataset.bioInsights_dir_cell_types+'Areas/'):
        os.mkdir(dataset.bioInsights_dir_cell_types+'Areas/')
    h_E_Fig.savefig(dataset.bioInsights_dir_cell_types + 'Areas/heatmap_Neighborhood_composition_of_Areas_Raw.png',dpi=600) 

    # Save heatmap Neigh_to_Pheno Normalized
    plt.close()
    plt.figure()
    sns.set(font_scale=1.1)
    row_colors_Area=cm.jet_r(range(0,255,int(255/clusters[2])))[:,:3]
    yticklabels_Area = ['A'+str(i+1) for i in range(clusters[2])]
    col_colors_Neigh=cm.jet_r(range(0,255,int(255/clusters[1])))[:,:3]
    xticklabels_Neigh = ['N'+str(i+1) for i in range(clusters[1])]
    Area_to_Neigh = stats.zscore(Area_to_Neigh,axis=0)
    Area_to_Neigh[np.isnan(Area_to_Neigh)] = 0
    h_E_Fig = sns.clustermap(Area_to_Neigh.transpose(),col_cluster=True, row_cluster=True, col_colors=row_colors_Area,
                                row_colors=col_colors_Neigh,yticklabels=xticklabels_Neigh,xticklabels=yticklabels_Area, linewidths=0.5, cmap="bwr")            
    h_E_Fig.savefig(dataset.bioInsights_dir_cell_types + 'Areas/heatmap_Neighborhood_composition_of_Areas_Norm.png',dpi=600) 
