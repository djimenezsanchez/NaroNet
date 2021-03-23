
import numpy as np
import copy
import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd

def save_celltype_abundance_per_patient(dataset, IndexAndClass, clusters, heatmapClusterPresence, labels):
    
    heatmap_Name = ['Phenotypes','Neighborhoods','Areas']
    ticklabel = ['P','N','A']

    for idx, ClusterLevel in enumerate(clusters):        
        
        list_of_labels = [i[0] for i in labels[str(ClusterLevel)]]

        # Save heatmap as excel file.
        dict_to_Save = {'Patient_Names':[ia[0] for ia in IndexAndClass],'Patient_Labels':list_of_labels}
        for TME in range(heatmapClusterPresence[str(ClusterLevel)].shape[1]):
            dict_to_Save[ticklabel[idx]+str(TME+1)]=list(heatmapClusterPresence[str(ClusterLevel)][:,TME])
        df1 = pd.DataFrame.from_dict(dict_to_Save)
        df1.to_excel(dataset.bioInsights_dir_cell_types_abundance+'heatmap_{}_Abundance.xlsx'.format(heatmap_Name[idx]))  

        # Save color information for next Cluster Level          
        labels_pal = sns.cubehelix_palette(len(set(list_of_labels)), light=.9, dark=.1, reverse=True, start=0, rot=-2)
        labels_lut = dict(zip(map(str, list(set(list_of_labels))), labels_pal))        
        labels_colors = [labels_lut[str(k)] for k in list_of_labels]
        
        # Create ClusterMap        
        heatmapClusterPresence_Fig = sns.clustermap(heatmapClusterPresence[str(ClusterLevel)],col_cluster=True, 
                                        row_colors=labels_colors, linewidths=0, cmap="vlag",
                                        xticklabels=[ticklabel[idx]+str(ind+1) for ind in range(ClusterLevel)])            
            
        # Save heatmap
        for label in list(set(list_of_labels)):
            heatmapClusterPresence_Fig.ax_col_dendrogram.bar(0, 0, color=labels_lut[str(label)], label=label, linewidth=0)
        heatmapClusterPresence_Fig.ax_col_dendrogram.legend(title=dataset.experiment_label[0])#, loc="left")#, ncol=5, bbox_to_anchor=(0.47, 0.8), bbox_transform=plt.gcf().transFigure)
        heatmapClusterPresence_Fig.savefig(dataset.bioInsights_dir_cell_types_abundance+'heatmap_{}_Abundance.png'.format(heatmap_Name[idx]),dpi=1000) 

def obtain_celltype_abundance_per_patient(dataset, IndexAndClass, clusters, ClusterThreshold):
    # Initialize heatmap Cluster Presence 
    if len(clusters)==1:
        heatmapClusterPresence = {str(clusters[0]):[]}        
    elif len(clusters)==2:
        heatmapClusterPresence = {str(clusters[0]):[], str(clusters[1]):[]}        
    elif len(clusters)==3:
        heatmapClusterPresence = {str(clusters[0]):[], str(clusters[1]):[], str(clusters[2]):[]}        
    labels = copy.deepcopy(heatmapClusterPresence)
    for idx, ClusterLevel in enumerate(clusters):
        heatmapClusterPresence[str(ClusterLevel)] = np.zeros([len(IndexAndClass),ClusterLevel])        
    
    # Obtain clusters per Slide
    for count, idxclster in enumerate(IndexAndClass):       
        # Obtain clusters per cluster Level
        for idx, ClusterLevel in enumerate(clusters):
            # Load Cluster Assignment for a specific image
            if idx<2:
                cluster_assignment_raw = np.load(dataset.processed_dir_cell_types+'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(idxclster[1],0, ClusterLevel))
                cluster_assignment_raw[np.percentile(cluster_assignment_raw.max(-1),ClusterThreshold)>cluster_assignment_raw]=1e-16
                cluster_assignment_raw = cluster_assignment_raw.sum(-2)                
            else:
                cluster_assignment_raw = np.load(dataset.processed_dir_cell_types+'cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(idxclster[1], ClusterLevel))                                            
            if idx>0:
                cluster_assignment_prev = copy.deepcopy(cluster_assignment)
                cluster_assignment = copy.deepcopy(cluster_assignment_raw)
            else:
                cluster_assignment = copy.deepcopy(cluster_assignment_raw)
            
            # Save value to heatmap
            if idx==0:
                heatmapClusterPresence[str(ClusterLevel)][count,:] = cluster_assignment_raw/(cluster_assignment_raw.sum()+1e-16)
            else:
                if len(cluster_assignment_raw.shape)==1:
                    heatmapClusterPresence[str(ClusterLevel)][count,:] = cluster_assignment_raw/(cluster_assignment_raw.sum()+1e-16)
                else:
                    clustersNOW=cluster_assignment_prev # Eliminate clusters with zeros attention from the previous iteration.
                    cluster_assignment = np.matmul(clustersNOW,cluster_assignment)
                    heatmapClusterPresence[str(ClusterLevel)][count,:] = (cluster_assignment/(cluster_assignment.sum()+1e-16))                        
            # Assign clusters to values            
            labels[str(ClusterLevel)].append(idxclster[2])
    return heatmapClusterPresence, labels
