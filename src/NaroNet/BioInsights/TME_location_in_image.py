import statistics as st
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from tifffile.tifffile import imwrite
import matplotlib.patches as mpatches
from PIL import Image
from NaroNet.utils.parallel_process import parallel_process

def TME_location_in_image_(dataset,thisfolder,idxclster,patchIDX, clusters,nClust,sts,statisticalTests_PerPatient,statisticalTests,unrestrictedLoss,count,Patindx):
    '''
    '''

    if not os.path.exists(dataset.bioInsights_dir_TME_in_image+thisfolder+idxclster[2][0]+'/'):
        os.mkdir(dataset.bioInsights_dir_TME_in_image+thisfolder+idxclster[2][0]+'/')

    # Pixel-to-cluster Matrix
    clust0 = np.load(dataset.processed_dir_cell_types+'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(idxclster[1], 0, clusters[0]))                    
    if nClust>0:
        clust1 = np.load(dataset.processed_dir_cell_types+'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(idxclster[1], 0, clusters[1]))                                     
    if nClust>1:
        clust2 = np.load(dataset.processed_dir_cell_types+'cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(idxclster[1], clusters[2]))                    
        clust2 = np.matmul(clust1,clust2)
    clust = clust0 
    clust = clust1 if nClust==1 else clust
    clust = clust2 if nClust==2 else clust                
    
    # Open Raw Image.
    im, imList = dataset.open_Raw_Image(idxclster,1)

    # Assign significant cluster as 1 to SuperPatch Image             
    if ('Endometrial_LowGrade' in dataset.raw_dir):
        for imList_n ,imList_i in enumerate(imList[::-1]):
            Patch_im_i = np.zeros(imList_i.shape[:2])                        
            # Create superPatch Label image using the Superpatch size. 
            division_rows = np.floor(Patch_im_i.shape[0]/dataset.patch_size)
            division_cols = np.floor(Patch_im_i.shape[1]/dataset.patch_size)
            lins = np.repeat(list(range(int(division_cols))), dataset.patch_size)
            lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
            for row_indx, row in enumerate(range(int(division_rows))):
                Patch_im_i[row_indx*dataset.patch_size:(row_indx+1)*dataset.patch_size,:int(division_cols*dataset.patch_size)] = np.transpose(lins+int(row_indx*division_cols))
            Patch_im_i = Patch_im_i.astype(int) 
            if imList_n==0:
                Patch_im = Patch_im_i
            else:                
                Patch_im = np.concatenate((Patch_im, Patch_im_i+Patch_im.max()+1),1)
    elif 'ZuriBasel' in dataset.raw_dir:
        for imList_n ,imList_i in enumerate(imList[::-1]):
            Patch_im_i = np.zeros(imList_i.shape[:2])                        
            # Create superPatch Label image using the Superpatch size. 
            division_rows = np.floor(Patch_im_i.shape[0]/dataset.patch_size)
            division_cols = np.floor(Patch_im_i.shape[1]/dataset.patch_size)
            lins = np.repeat(list(range(int(division_cols))), dataset.patch_size)
            lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
            for row_indx, row in enumerate(range(int(division_rows))):
                Patch_im_i[row_indx*dataset.patch_size:(row_indx+1)*dataset.patch_size,:int(division_cols*dataset.patch_size)] = np.transpose(lins+int(row_indx*division_cols))
            Patch_im_i = Patch_im_i.astype(int) 
            if imList_n==0:
                Patch_im = Patch_im_i
            else:                
                Patch_im = np.concatenate((Patch_im, Patch_im_i+Patch_im.max()+1),1)
    else:
        Patch_im = np.zeros(im.shape[:2])                        
        # Create superPatch Label image using the Superpatch size. 
        division_rows = np.floor(Patch_im.shape[0]/dataset.patch_size)
        division_cols = np.floor(Patch_im.shape[1]/dataset.patch_size)
        lins = np.repeat(list(range(int(division_cols))), dataset.patch_size)
        lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
        for row_indx, row in enumerate(range(int(division_rows))):
            Patch_im[row_indx*dataset.patch_size:(row_indx+1)*dataset.patch_size,:int(division_cols*dataset.patch_size)] = np.transpose(lins+int(row_indx*division_cols))
        # suprpxlIm = np.transpose(suprpxlIm)
        Patch_im = Patch_im.astype(int)
        if ('V_H' in dataset.root) or ('V3' in dataset.root) or ('V1' in dataset.root) or ('V4' in dataset.root):
            Patch_im = np.transpose(Patch_im)                        

    # Assign significant clusters to pixels.
    cell_type_top1 = np.apply_along_axis(np.argmax, axis=1, arr=clust)       

    # Obtain Image in RGB
    # imRGB = dataset.nPlex2RGB(im)                                   
    # for c in range(imRGB.shape[2]):
    #     imRGB[:,:,c] = imRGB[:,:,c]/imRGB[:,:,c].max()                 
                            
    # For each threshold value
    #for thrs in ClusteringThreshold:
    # Mask selecting top PIR values.
    AllClusters = (copy.deepcopy(cell_type_top1)*0).astype('float32')
    AllClusters[sts[2][1]==cell_type_top1] += clust.max(-1)[sts[2][1]==cell_type_top1]
    # Apply Threshold to select cells with high confidence.
    # AllClusters[np.percentile(clust.max(-1),thrs)>clust.max(-1)] = 0 
    AllClusters_2=AllClusters[Patch_im] # Superpatches of significant clusters.
    # Number of SuperPatches present in this one    
    # Save Certainty of this Phenotype-TissueCommunity                        
    plt.figure()
    n, bins, patches = plt.hist(clust.max(-1)[sts[2][1]==cell_type_top1], 100, color=cm.jet_r(int(sts[2][1]*(255/int(sts[2][0])))), alpha=0.5)
    if int(sts[2][0])==clusters[0]:
        plt.xlabel('Level of Phenotype Certainty')
        plt.title('Histogram of Phenotype '+str(sts[2][1])+' Certainty')
    elif int(sts[2][0])==clusters[1]:
        plt.xlabel('Level of Neighborhood Certainty')
        plt.title('Histogram of Neighborhood '+str(sts[2][1])+' Certainty')
    elif int(sts[2][0])==clusters[2]:
        plt.xlabel('Level of Area Certainty')
        plt.title('Histogram of Area '+str(sts[1])+' Certainty')
    plt.ylabel('Number of patches')
    #plt.axvline(x=np.percentile(clust.max(-1),thrs), color='r', linestyle='dashed', linewidth=2)
    plt.savefig(dataset.bioInsights_dir_TME_in_image+thisfolder+idxclster[2][0]+'/Label{}_Slide{}_Patch{}_Clstrs{}_Thrs{}_Acc{}_PIR{}_Hist.png'.format(
        idxclster[2][0],idxclster[0],patchIDX,statisticalTests['TME -h'][count],100,unrestrictedLoss[count],str(round(statisticalTests_PerPatient[Patindx][1],2))), format="PNG",dpi=200) 
    # Save GT and Image                           
    imwrite(dataset.bioInsights_dir_TME_in_image+thisfolder+idxclster[2][0]+'/Label{}_Slide{}_Patch{}_Clstrs{}_Acc{}_PIR{}_Images.tiff'.format(idxclster[2][0],idxclster[0],patchIDX,statisticalTests['TME -h'][count],unrestrictedLoss[count],str(round(statisticalTests_PerPatient[Patindx][1],2))),np.moveaxis(im,2,0))                                        
    imtosave = Image.fromarray(np.uint8(AllClusters_2*255))
    imtosave.save(dataset.bioInsights_dir_TME_in_image+thisfolder+idxclster[2][0]+'/Label{}_Slide{}_Patch{}_Clstrs{}_Acc{}_PIR{}_Label.tiff'.format(idxclster[2][0],idxclster[0],patchIDX,statisticalTests['TME -h'][count],unrestrictedLoss[count],str(round(statisticalTests_PerPatient[Patindx][1],2))))                
    return 'done'

def TME_location_in_image(dataset, statisticalTests, clusters, IndexAndClass,unrestrictedLoss,statisticalTests_PerPatient, num_classes, attentionLayer,ClusteringThreshold):
    '''
        docstring
    '''
    
    # statisticalTests = sorted(statisticalTests, key=lambda k: k[0]) # 1.p-value, 2.Cluster step, 3. column of the heatmap        
    # stsTest=statisticalTests[0]
    IntersecIndex=[]
    patchIDX = dataset.args['epochs']
    dict_subjects = []
    for nClust, clusterToProcess in enumerate(clusters):            
        IntersecIndex.append([])
        # For each image...
        for count, Patindx in enumerate(statisticalTests['Patient index']):      
            idxclster = IndexAndClass[Patindx]                   
            thisfolder = statisticalTests['TME -h'][count]+'/'
            if not os.path.exists(dataset.bioInsights_dir_TME_in_image+thisfolder):
                os.mkdir(dataset.bioInsights_dir_TME_in_image+thisfolder)            
            # One Phenotype or Tissue Community to one image
            for sts in statisticalTests['TME'][count]:
                if int(sts[2][0])==clusterToProcess: 
                    dict_subjects.append({'dataset':dataset,'thisfolder':thisfolder,'idxclster':idxclster,'patchIDX':patchIDX, 'clusters':clusters,'nClust':nClust,'sts':sts,'statisticalTests_PerPatient':statisticalTests_PerPatient,'statisticalTests':statisticalTests,'unrestrictedLoss':unrestrictedLoss,'count':count,'Patindx':Patindx})
    
    # select_patches_from_cohort
    result = parallel_process(dict_subjects,TME_location_in_image_,use_kwargs=True,front_num=2,desc='BioInsights: Save TOP-k PIRs for each TME') 
    return 1

def All_TMEs_in_Image_(clusters,dataset,subject_info,ClusteringThreshold,thisfolder):
    # Display Interaction-graph for each cluster_level
    for idx, n_cell_types in enumerate(clusters):                                
        # Apply mask to patch
        for subgraph_idx in range(dataset.findLastIndex(subject_info[1])+1):                
            # Open Raw Image.
            im, imList = dataset.open_Raw_Image(subject_info,1)
            
            # Obtain patch_clustering                
            if idx<2:
                clust = np.load(dataset.processed_dir_cell_types+'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(subject_info[1], subgraph_idx, n_cell_types))                    
                clust[np.percentile(clust.max(-1),ClusteringThreshold)>clust] = 1e-16

            # for 2nd cluster obtain cluster_assignment per node.
            if idx>1:
                # Load clusters of this iteration
                cluster_assignment = np.load(dataset.processed_dir_cell_types+'cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(subject_info[1], n_cell_types))                    
                # Obtain clusters of clusters.
                clust = np.matmul(clust_prev,cluster_assignment)
            clust_prev = copy.deepcopy(clust)

            # Obtain Superpixel Image
            if 'Endometrial_LowGrade' in dataset.root:                        
                for imList_n ,imList_i in enumerate(imList):
                    Patch_im_i = np.zeros(imList_i.shape[:2])                        
                    # Create superPatch Label image using the Superpatch size. 
                    division = np.floor(Patch_im_i.shape[0]/dataset.patch_size)
                    lins = np.repeat(list(range(int(division))), dataset.patch_size)
                    lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
                    for y_indx, y in enumerate(range(int(division))):
                        Patch_im_i[:int(division*dataset.patch_size),y_indx*dataset.patch_size:(y_indx+1)*dataset.patch_size] = lins+int(y_indx*division)
                    Patch_im_i = np.transpose(Patch_im_i)
                    Patch_im_i = Patch_im_i.astype(int) 
                    if imList_n==0:
                        Patch_im = Patch_im_i
                    else:
                        Patch_im_i = Patch_im
                        Patch_im = np.concatenate((Patch_im, Patch_im_i+Patch_im.max()+1),1)
                Patch_im=Patch_im+1
            elif 'Endometrial_POLE' in dataset.root:
                Patch_im = np.zeros(im.shape[:2])                        
                # Create superPatch Label image using the Superpatch size. 
                division_rows = np.floor(Patch_im.shape[0]/dataset.patch_size)
                division_cols = np.floor(Patch_im.shape[1]/dataset.patch_size)
                lins = np.repeat(list(range(int(division_cols))), dataset.patch_size)
                lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
                for row_indx, row in enumerate(range(int(division_rows))):
                    Patch_im[row_indx*dataset.patch_size:(row_indx+1)*dataset.patch_size,:int(division_cols*dataset.patch_size)] = np.transpose(lins+int(row_indx*division_cols))
                # Patch_im = np.transpose(Patch_im)
                Patch_im = Patch_im.astype(int)

            elif 'Synthetic' in dataset.root:
                # if 'Superpixel' in dataset.raw_dir:
                #     Patch_im = np.load(osp.join(dataset.root,'OriginalSuperpixel','{}.npy'.format('Labels_'+subject_info[0][11:])))
                # elif 'SuperPatch' in dataset.raw_dir:
                Patch_im = np.zeros(im.shape[:2])                        
                # Create superPatch Label image using the Superpatch size. 
                division = np.floor(Patch_im.shape[0]/dataset.patch_size)
                lins = np.repeat(list(range(int(division))), dataset.patch_size)
                lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
                for y_indx, y in enumerate(range(int(division))):
                    Patch_im[:int(division*dataset.patch_size),y_indx*dataset.patch_size:(y_indx+1)*dataset.patch_size] = lins+int(y_indx*division)
                if ('V2' in dataset.root)  or ('V_H' in dataset.root)  or ('V4' in dataset.root):
                    Patch_im = np.transpose(Patch_im)
                Patch_im = Patch_im.astype(int)            
            elif 'Images-Cytof52Breast' in dataset.root:
                Patch_im = np.load(osp.join(dataset.root,'Original','{}.npy'.format(subject_info[0]+'Labels')))     
            elif 'Lung' in dataset.root:
                Patch_im = np.zeros(im.shape[:2])                        
                # Create superPatch Label image using the Superpatch size. 
                division = np.floor(Patch_im.shape[0]/dataset.patch_size)
                lins = np.repeat(list(range(int(division))), dataset.patch_size)
                lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
                for y_indx, y in enumerate(range(int(division))):
                    Patch_im[:int(division*dataset.patch_size),y_indx*dataset.patch_size:(y_indx+1)*dataset.patch_size] = lins+int(y_indx*division)
                Patch_im = Patch_im.astype(int)
                Patch_im = np.transpose(Patch_im)
            elif ('ZuriBasel' in dataset.root) or True:
                # if 'Superpixel' in dataset.raw_dir:
                #     Patch_im = np.load(osp.join(dataset.root,'OriginalSuperpixel','{}.npy'.format('Labels_'+subject_info[0][11:])))
                # elif 'SuperPatch' in dataset.raw_dir:
                # Create Mosaic                            
                x_dim = max([i.shape[0] for i in imList])
                y_dim = sum([i.shape[1] for i in imList])
                Patch_im = np.zeros((x_dim,y_dim))
                last_y = 0
                
                for imList_n ,imList_i in enumerate(imList):
                    Patch_im_i = np.zeros(imList_i.shape[:2])                        
                    # Create superPatch Label image using the Superpatch size. 
                    division = np.floor(Patch_im_i.shape[0]/dataset.patch_size)
                    lins = np.repeat(list(range(int(division))), dataset.patch_size)
                    lins = np.repeat(np.expand_dims(lins,axis=1), dataset.patch_size, axis=1)
                    for y_indx, y in enumerate(range(int(np.floor(Patch_im_i.shape[1]/dataset.patch_size)))):
                        Patch_im_i[:int(division*dataset.patch_size),y_indx*dataset.patch_size:(y_indx+1)*dataset.patch_size] = lins+int(y_indx*division)
                    # Patch_im_i = np.transpose(Patch_im_i)
                    Patch_im_i = Patch_im_i.astype(int) 
                    if len(imList)==0:
                        Patch_im = Patch_im_i
                    else:                                                                   
                        Patch_im[:Patch_im_i.shape[0],last_y:last_y+Patch_im_i.shape[1]] = Patch_im_i[:,:]
                        last_y = last_y + Patch_im_i.shape[1]           
                Patch_im=Patch_im+1
                Patch_im = Patch_im.astype(int)

            # Assign superpixels to Clusters.                    
            CLST_Patch_im = np.zeros((Patch_im.shape[0],Patch_im.shape[1],4),dtype=float)
            CLST_suprpxlVal = np.zeros((Patch_im.shape[0],Patch_im.shape[1]),dtype=int)
            cell_type_top1 = np.apply_along_axis( np.argmax, axis=1, arr=clust)                    
            CLST_Patch_im = cm.jet_r((cell_type_top1[Patch_im-1]*(255/clust.shape[-1])).astype(int)) # Jet_r values goes from 0 to 256
            CLST_suprpxlVal = cell_type_top1[Patch_im-1].astype(int)
            plt.figure()#figsize=(8,4))

            # Obtain multitiff label image 
            im_list = []
            im_multi = cell_type_top1[Patch_im-1]
            for TME_Id in range(n_cell_types+1):
                im_list.append((im_multi==TME_Id)*1)
            im_list = np.stack(im_list)
            
            
            # ax = plt.Axes(fig, [0., 0., 1., 1.])
            # ax.set_axis_off()
            # fig.add_axes(ax)
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            iim = plt.imshow(CLST_Patch_im, interpolation='none')
            colors = cm.jet_r(((255/clust.shape[-1])*np.array(list(range(clust.shape[-1])))).astype(int))
            # create a patch (proxy artist) for every color 
            patches = [ mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=i_n) ) for i_n,i in enumerate(range(len(colors))) ]
            # put those patched as legend-handles into the legend
            plt.axis('off')
            plt.figure()
            plt.legend(handles=patches)
            plt.axis('off')
            # axim = ax.imshow(CLST_Patch_im)
            # fig.colorbar(axim, ax=ax,ticks=np.linspace(0,255,clust.shape[-1]))
            if idx==0:
                plt.imsave(dataset.bioInsights_dir_TME_in_image+thisfolder+'Phenotypes/TMEs_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}_Label.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),CLST_Patch_im)                
                imwrite(dataset.bioInsights_dir_TME_in_image+thisfolder+'Phenotypes/TMEs_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}_Images.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),np.moveaxis(im,2,0))                                        
                plt.savefig(dataset.bioInsights_dir_TME_in_image+thisfolder+'Phenotypes/TMEsLegend_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),dpi=300)
            elif idx==1:
                plt.imsave(dataset.bioInsights_dir_TME_in_image+thisfolder+'Neighborhoods/TMEs_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}_Label.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),CLST_Patch_im)
                imwrite(dataset.bioInsights_dir_TME_in_image+thisfolder+'Neighborhoods/TMEs_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}_Images.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),np.moveaxis(im,2,0))                                                        
                plt.savefig(dataset.bioInsights_dir_TME_in_image+thisfolder+'Neighborhoods/TMEsLegend_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),dpi=300)
            else:
                plt.imsave(dataset.bioInsights_dir_TME_in_image+thisfolder+'Areas/TMEs_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}_Label.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),CLST_Patch_im)
                imwrite(dataset.bioInsights_dir_TME_in_image+thisfolder+'Areas/TMEs_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}_Images.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),np.moveaxis(im,2,0))                                                        
                plt.savefig(dataset.bioInsights_dir_TME_in_image+thisfolder+'Areas/TMEsLegend_{}_Slide{}_Patch{}_Clust{}_iter{}_Thr{}.tiff'.format(subject_info[0],subject_info[1],subgraph_idx,n_cell_types,dataset.args['epochs'],ClusteringThreshold),dpi=300)   


def All_TMEs_in_Image(dataset,clusters, statistics_per_patient, IndexAndClass, num_classes, attentionLayer,ClusteringThreshold):
    # Apply mask to slide
    lastidx=0 
    cluster_assignment_attn=None
    dict_subjects = []
    for count, subject_info in enumerate(IndexAndClass[pi] for pi in statistics_per_patient['Patient index']): 
        if not os.path.exists(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count]+'/'):
            os.mkdir(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count])           
        if not os.path.exists(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count]+'/Phenotypes/'):
            os.mkdir(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count]+'/Phenotypes/')
        if not os.path.exists(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count]+'/Neighborhoods/'):
            os.mkdir(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count]+'/Neighborhoods/')
        if not os.path.exists(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count]+'/Areas/'):
            os.mkdir(dataset.bioInsights_dir_TME_in_image+statistics_per_patient['TME -h'][count]+'/Areas/')
        thisfolder = statistics_per_patient['TME -h'][count]+'/'

        dict_subjects.append({'clusters':clusters,'dataset':dataset,'subject_info':subject_info,'ClusteringThreshold':ClusteringThreshold,'thisfolder':thisfolder})
    
    result = parallel_process(dict_subjects,All_TMEs_in_Image_,use_kwargs=True,front_num=5,desc='BioInsights: Save TMEs for each patient') 
