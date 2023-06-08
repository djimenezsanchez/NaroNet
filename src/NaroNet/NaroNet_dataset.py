import os
import os.path as osp

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Dataset, Data
import NaroNet.utils.graph_data as graph_data
from torch_geometric.data.makedirs import makedirs
import numpy as np
from torch_geometric.nn import radius_graph
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib as mpl
from skimage import io
import xlrd
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import copy
import itertools
# import statsmodels as sm
# import statsmodels.api as smapi
import csv
import statistics as st
from tqdm import tqdm
import NaroNet.utils.utilz as utilz
from tifffile.tifffile import imwrite

from sklearn.manifold import TSNE
import pandas
import random
# from imread import imread, imsave
import tifffile
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.cluster.hierarchy import fcluster
import scipy.io as sio
from torch_scatter import scatter_add
import matplotlib.patches as mpatches
from PIL import Image as pil
import pandas as pd
from NaroNet.BioInsights.Pheno_Neigh_Info import load_cell_types_assignments
from NaroNet.BioInsights.Pheno_Neigh_Info import load_patch_image
from NaroNet.BioInsights.Pheno_Neigh_Info import topk_confident_patches
from NaroNet.BioInsights.Pheno_Neigh_Info import extract_topk_patches_from_cohort
from NaroNet.BioInsights.Pheno_Neigh_Info import save_heatMapMarker_and_barplot
from NaroNet.BioInsights.Pheno_Neigh_Info import obtain_neighborhood_composition
from NaroNet.BioInsights.Pheno_Neigh_Info import select_patches_from_cohort
from NaroNet.BioInsights.Cell_type_abundance import obtain_celltype_abundance_per_patient
from NaroNet.BioInsights.Cell_type_abundance import save_celltype_abundance_per_patient
from NaroNet.BioInsights.Differential_abundance_analysis import differential_abundance_analysis
from NaroNet.BioInsights.TME_location_in_image import TME_location_in_image
from NaroNet.BioInsights.TME_location_in_image import All_TMEs_in_Image
from NaroNet.BioInsights.Pheno_Neigh_Info import Area_to_Neighborhood_to_Phenotype
from NaroNet.BioInsights.Synthetic_GT_TME import ObtainMultivariateIntersectInSynthetic
import NaroNet

import cv2
import networkx as nx 

class NARODataset(torch.utils.data.Dataset):
    r"""Dataset base class for creating graph datasets.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html>`__ for the accompanying tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __len__(self):
        r"""The number of examples in the dataset."""
        raise NotImplementedError

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    def __init__(self,
                 root,
                 patch_size,
                 recalculate,
                 UseSuperpatch,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(NARODataset, self).__init__()

        self.root = osp.expanduser(osp.normpath(root))
        if UseSuperpatch:
            self.raw_dir = osp.join(osp.join(self.root,'Patch_Contrastive_Learning'),'Image_Patch_Representation')
        else:
            self.raw_dir = osp.join(osp.join(self.root,'Patch_Contrastive_Learning'),'Image_Patch_Representation')

        self.processed_dir = osp.join(osp.join(self.root,'NaroNet'),'Enriched_graph_'+str(patch_size+1))
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.context_size = patch_size+1
        self.recalculate = recalculate
        self._process()

    @property
    def num_node_features(self):
        data = get(0)
        return data.num

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the dataset."""
        return self[0].num_edge_features

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def __getitem__(self, idx):  # pragma: no cover
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given)."""
        data = self.get(idx)
        data = data if self.transform is None else self.transform(data)
        return data

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

class NaroNet_dataset(torch.utils.data.Dataset):
    '''
    Dataset base class for creating graph datasets.
    '''
    
    def __init__(self, root, patch_size, transform=None, pre_transform=None, recalculate=False, superPatchEmbedding=None, experiment_label=None):
                
        self.patch_size = patch_size
        self.SuperPatchEmbedding = superPatchEmbedding
        self.root = root#osp.expanduser(osp.normpath(root))
        self.transform = transform
        self.context_size = patch_size+1
        self.recalculate = recalculate
        self.experiment_label = experiment_label
        self.TrainingClusterMapEpoch=-1

        self.raw_dir = osp.join(osp.join(self.root,'Patch_Contrastive_Learning'),'Image_Patch_Representation')
        self.processed_dir = osp.join(osp.join(self.root,'NaroNet'),'_'.join(experiment_label))
        self.processed_dir_graphs = self.processed_dir+'/Subject_graphs/'
        self.processed_dir_cross_validation = self.processed_dir+'/Cross_validation_results/'
        self.processed_dir_cell_types = self.processed_dir+'/Cell_type_assignment/'
        self.bioInsights_dir = self.root+'BioInsights/'+'_'.join(experiment_label)+'/'
        self.bioInsights_dir_cell_types =  self.bioInsights_dir+'Cell_type_characterization/'
        self.bioInsights_dir_cell_types_Neigh =  self.bioInsights_dir_cell_types+'Neighborhoods/'
        self.bioInsights_dir_cell_types_Pheno =  self.bioInsights_dir_cell_types+'Phenotypes/'
        self.bioInsights_dir_cell_types_abundance =  self.bioInsights_dir+'Cell_type_abundance/'
        self.bioInsights_dir_abundance_analysis =  self.bioInsights_dir+'Differential_abundance_analysis/'
        self.bioInsights_dir_abundance_analysis_global = self.bioInsights_dir_abundance_analysis+'GlobalAnalysis/'
        self.bioInsights_dir_TME_in_image =  self.bioInsights_dir+'Locate_TME_in_image/'                
        self.bioInsights_dir_abundance_analysis_Subgroups =  self.bioInsights_dir_abundance_analysis + 'Inter_Intra_Patient_Analysis/'

        makedirs(self.processed_dir)
        makedirs(self.processed_dir_graphs)
        makedirs(self.processed_dir_cross_validation)
        makedirs(self.processed_dir_cell_types)
        makedirs(self.bioInsights_dir)
        makedirs(self.bioInsights_dir_cell_types)
        makedirs(self.bioInsights_dir_cell_types_Neigh)
        makedirs(self.bioInsights_dir_cell_types_Pheno)
        makedirs(self.bioInsights_dir_cell_types_abundance)
        makedirs(self.bioInsights_dir_abundance_analysis)
        makedirs(self.bioInsights_dir_abundance_analysis_global)
        makedirs(self.bioInsights_dir_abundance_analysis_Subgroups)
        makedirs(self.bioInsights_dir_TME_in_image)        
        self.process()


    @property
    def processed_file_names(self):
        a = [name.split('_')[1] for name in os.listdir(self.processed_dir_graphs) if 'data' in name]
        b = [int(value) for value in a if not '.pt' in value]        
        return max(b)

    def __len__(self):
        return self.processed_file_names

    def findLastIndex(self,idx):
        lastidx=0
        for root, dirs, files in os.walk(self.processed_dir):
            for file in files:
                if "_"+str(idx)+"_" in file and '.pt' in file:
                    idxNow = int(file.split('_')[2].split('.')[0])
                    if idxNow>lastidx: lastidx = idxNow                     
        return lastidx

    def get(self, idx, subIm):                
        return torch.load(osp.join(self.processed_dir_graphs, 'data_{}_{}.pt'.format(idx,subIm)))

    def getStatistics(self, size):
        # We Obtain the last file with data because it has all the information needed to build the Neural Network
        last_data = self.get(size,self.findLastIndex(size))#dataset.__len__() 
        random.shuffle(last_data.IndexAndClass)
        return last_data.num_total_nodes, last_data.edge_index_total, last_data.num_features, last_data.mean_STD, last_data.IndexAndClass, last_data.Percentile, last_data.num_classes, last_data.name_labels

    def normalizeAdjacency(self,edge_index,num_total_nodes):
        # Normalize edgeIndex            
        row, col = edge_index
        edge_weight = torch.ones((edge_index.size(1), ),device='cpu')
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_total_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def gatherOneDataAugment(self,num_total_nodes,num_features,dataNOW,countIter,device,name_labels,y_ALL,x_ALL,edge_index_ALL,num_nodes,i, model_type, k_hops, dataAugmentationPerc):
        # Prepare data in a Tensor to be readable by the Neural Network
        # Initialize tensors
        x = torch.zeros(num_total_nodes,num_features,dtype=torch.float32)        
        # Assign label to tensor
        y = [i.index(dataNOW.y[n]) for n,i in enumerate(name_labels)]  
        y = torch.from_numpy(np.array(y))       
        # Data Augmentation - Apply Gaussian Noise to nodes
        noise = torch.from_numpy(np.random.normal(loc=0, scale=(abs(dataNOW.x.min())+dataNOW.x.max()).item()*dataAugmentationPerc/2, size=x.shape))
        x += noise
        # Assign Features to tensor
        x[:dataNOW.x.shape[0],:dataNOW.x.shape[1]]+=dataNOW.x[:dataNOW.x.shape[0],:dataNOW.x.shape[1]]        
        # Data Augmentation - Drop nodes
        x[random.choices(range(dataNOW.x.shape[0]), k=int(dataAugmentationPerc*dataNOW.x.shape[0])),:] = 0
        # Data Augmentation - Drop one channel from certain nodes
        for feat in random.choices(range(dataNOW.x.shape[1]), k=3): # Iterate over features
            x[random.choices(range(dataNOW.x.shape[0]), k=int(dataAugmentationPerc*dataNOW.x.shape[0])), feat] = 0                    
        x=x.unsqueeze(0)

        # Create Dense matrix from sparse connections.
        countIter += 1
        normalizeAdjMat = True
        if normalizeAdjMat:
            # Normalize Adjacency matrix.
            norm = self.normalizeAdjacency(dataNOW.edge_index,num_total_nodes)
            # Data Augmentation - Drop Edges
            norm[random.choices(range(norm.shape[0]), k=int(dataAugmentationPerc*norm.shape[0]))]=0
        else:
            norm =  torch.ones((dataNOW.edge_index.size(1), ),device='cpu')
        # Obtain Sparse Adjacency Matrix
        edge_index = torch.sparse.FloatTensor(dataNOW.edge_index, norm, torch.Size([num_total_nodes,num_total_nodes]))
        edge_index = edge_index.to(device)
        
        # Concatenate actual patch to the rest of the batch.
        if len(y_ALL)==0:
            y_ALL=y
            x_ALL=x
            edge_index_ALL=[edge_index]
            num_nodes = [dataNOW.num_nodes]
        else:
            y_ALL=torch.cat((y_ALL,y),0)
            x_ALL=torch.cat((x_ALL,x),0)
            edge_index_ALL.append(edge_index)
            num_nodes.append(dataNOW.num_nodes)
        return countIter, y_ALL, x_ALL, edge_index_ALL, num_nodes


    def generateEmptyGraph(self):
        dataNOW = self.get(0,0)
        dataNOW.edge_index = torch.tensor([[0],[0]])
        dataNOW.x=torch.zeros([1,dataNOW.x.shape[1]]) 
        dataNOW.num_nodes = 0
        return dataNOW

    def generateEmptyClusteringGraph(self,batch_size,clusters,hidden):
        dataNOW = self.get(0,0)
        dataNOW.edge_index = torch.zeros([batch_size,clusters,clusters])
        dataNOW.x=torch.zeros([batch_size,clusters,hidden]) 
        dataNOW.num_nodes = 0
        dataNOW.y=[]
        return dataNOW

    def gatherData(self, args, indices,choosesubImFromHere,training):
        countIter=0
        edge_index_toadj_ALL=[]
        y_ALL=[]
        x_ALL=[]
        num_nodes=[]
        edge_index_ALL=[]
        savesubIm=[]
        # Load one patch from a specific slide.
        for count,index in enumerate(indices):
            if choosesubImFromHere[count]:
                subIm = random.sample(choosesubImFromHere[count],1) 
                savesubIm.append([subIm,count])                 
                choosesubImFromHere[count].remove(subIm[0])
                dataNOW = self.get(index,subIm[0])
                # Data augmentation when training
                if training:
                    countIter, y_ALL, x_ALL, edge_index_ALL, num_nodes = self.gatherOneDataAugment(args.num_total_nodes,args.num_features,dataNOW,countIter,args.device,args.name_labels,y_ALL,x_ALL,edge_index_ALL,num_nodes,index,args.args['modeltype'],args.args['n-hops'], args.args['dataAugmentationPerc'])
                else: 
                    countIter, y_ALL, x_ALL, edge_index_ALL, num_nodes = self.gatherOneDataAugment(args.num_total_nodes,args.num_features,dataNOW,countIter,args.device,args.name_labels,y_ALL,x_ALL,edge_index_ALL,num_nodes,index,args.args['modeltype'],args.args['n-hops'],0)

        # Normalize z-score and 99Percentile
        if args.args['normalizePercentile']:
            for n in range(x_ALL.shape[0]):
                aux=x_ALL[n,:,:]
                aux[aux>args.percentile[n]]=args.percentile[n]
                x_ALL[n,:,:]=aux                
        if args.args['normalizeFeats']:
            x_ALL = (x_ALL-args.mean_STD[0].float())/(args.mean_STD[1].float()+1e-16)
        if args.args['normalizeCells']:
            x_ALL=(x_ALL-x_ALL.mean(2).unsqueeze(2).repeat(1,1,x_ALL.shape[2]))/(x_ALL.std(2).unsqueeze(2).repeat(1,1,x_ALL.shape[2])+1e-16)

        return graph_data.Data(edge_index=edge_index_ALL,y=y_ALL,x=x_ALL, num_nodes=num_nodes), choosesubImFromHere, savesubIm

    def gatherData_UnsupContrast(self, args, indices,choosesubImFromHere,training):
        countIter=0
        edge_index_toadj_ALL=[]
        y_ALL=[]
        x_ALL=[]
        num_nodes=[]
        edge_index_ALL=[]
        savesubIm=[]
        # Load one patch from a specific slide.
        for count,index in enumerate(indices):
            if choosesubImFromHere[count]:
                subIm = random.sample(choosesubImFromHere[count],1) 
                savesubIm.append([subIm,count])                 
                choosesubImFromHere[count].remove(subIm[0])
                dataNOW = self.get(index,subIm[0])
                # Data augmentation when training
                countIter, y_ALL, x_ALL, edge_index_ALL, num_nodes = self.gatherOneDataAugment(args.num_total_nodes,args.num_features,dataNOW,countIter,args.device,indices,y_ALL,x_ALL,edge_index_ALL,num_nodes,index,args.args['modeltype'],args.args['n-hops'], args.args['dataAugmentationPerc'])
                countIter, y_ALL, x_ALL, edge_index_ALL, num_nodes = self.gatherOneDataAugment(args.num_total_nodes,args.num_features,dataNOW,countIter,args.device,indices,y_ALL,x_ALL,edge_index_ALL,num_nodes,index,args.args['modeltype'],args.args['n-hops'], 0)                

        # Normalize z-score and 99Percentile
        if args.args['normalizePercentile']:
            for n in range(x_ALL.shape[0]):
                aux=x_ALL[n,:,:]
                aux[aux>args.percentile[n]]=args.percentile[n]
                x_ALL[n,:,:]=aux                
        if args.args['normalizeFeats']:
            x_ALL = (x_ALL-args.mean_STD[0].float())/(args.mean_STD[1].float()+1e-16)
        if args.args['normalizeCells']:
            x_ALL=(x_ALL-x_ALL.mean(2).unsqueeze(2).repeat(1,1,x_ALL.shape[2]))/(x_ALL.std(2).unsqueeze(2).repeat(1,1,x_ALL.shape[2])+1e-16)
        
        return graph_data.Data(edge_index=edge_index_ALL,y=y_ALL,x=x_ALL, num_nodes=num_nodes), choosesubImFromHere, savesubIm


    def process(self):
        # Initial Value        
        num_total_nodes=0
        edge_index_total=0
        GraphIndex=0
        IndexAndClass=[]     

        if len(os.listdir(self.raw_dir))==len(os.listdir(self.processed_dir_graphs)):            
            return

        for root, dirs, files in os.walk(self.raw_dir):
            files.sort()
            for file_index in tqdm(range(len(files)),ascii=True,desc='NaroNet: generating enriched graphs'):
                fullpath = (osp.join(self.raw_dir,files[file_index]))                                
                    
                # Handle Superpatches npy files
                if '.npy' in fullpath:
                    file = np.load(fullpath,allow_pickle=True)
                else:
                    continue
                    
                # Obtain edge List
                maxValues = file[:,[0,1]].max(-2)+1
                minValues = file[:,[0,1]].min(-2)-1

                # Obtain Labels from excel
                patient_to_image_excel = pd.read_excel(self.root+'Raw_Data/Experiment_Information/Image_Labels.xlsx')  
                if '.' in str(patient_to_image_excel['Image_Names'][0]):
                    patient_to_image_excel['Image_Names'] = ['.'.join(i.split('.')[:-1]) for i in patient_to_image_excel['Image_Names']]
                patient_to_image_excel['Image_Names'] = [str(i) for i in patient_to_image_excel['Image_Names']]
                # Find the actual patient
                if '.'.join(files[file_index].split('.')[:-1]) in list(patient_to_image_excel['Image_Names']):
                    patient_index = list(patient_to_image_excel['Image_Names']).index('.'.join(files[file_index].split('.')[:-1]))                
                else:
                    continue

                # Select the respective patient's label
                patient_label = []
                for l in self.experiment_label:
                    patient_label.append(patient_to_image_excel[l][patient_index])                
                
                # Definition of grid size to extract patches. The graph could be separated into different subgraphs. Deactivated for now.
                ImageSize = 100000
                SubImageIndex=0

                # Extract patches from image
                for indexX in range(int(minValues[0]),int(maxValues[0])+ImageSize,ImageSize):
                    for indexY in range(int(minValues[1]),int(maxValues[1])+ImageSize,ImageSize):
                        
                        # Extract cells within limits of this iteration
                        Truex = np.logical_and((indexX+ImageSize)>file[:,[0]], file[:,[0]]>indexX)
                        Truey = np.logical_and((indexY+ImageSize)>file[:,[1]], file[:,[1]]>indexY)

                        # If there are no patches in the specified subregion go to the next one.
                        Truexy = np.logical_and(Truey,Truex)
                        if sum(Truexy==True)<2:
                            continue

                        # Generate graph using selected cells                        
                        edge_index = radius_graph(x=torch.tensor(file[np.squeeze(Truexy),:][:,[0,1]]), r=self.context_size, loop=True, max_num_neighbors=100)  
                        
                        # Avoid repeated patches.
                        edge_index = torch.sort(edge_index,dim=0).values
                        edge_index = torch.unique_consecutive(edge_index,dim=1)

                        #  Mean node degree
                        print('Mean Node Degree:'+str(edge_index.shape[1]/sum(Truexy==True))+' '+files[file_index])
                                                    
                        # Assign node attributes and label.
                        x = torch.from_numpy(file[np.squeeze(Truexy),:][:,2:])                                                
                        y = patient_label
                       
                        # Obtain data format for pytorch-geometric.
                        data = Data(edge_index=edge_index, x=x, y=y, name=files[file_index])


                        # Obtain the number of maximum nodes, so far.
                        if num_total_nodes<data.num_nodes:
                            num_total_nodes=data.num_nodes
                        data.num_total_nodes=num_total_nodes

                        # Obtain the number of maximum edges, so far.
                        if edge_index_total<data.edge_index.shape[1]:
                            edge_index_total=data.edge_index.shape[1]
                        data.edge_index_total=edge_index_total
                                                
                        # data.num_classes = data.y.shape
                        
                        # Add List with indices and labels.
                        if SubImageIndex==0:
                            IndexAndClass.append([files[file_index][:-4],GraphIndex,y]) # Save the first index of each class
                            data.IndexAndClass = IndexAndClass

                        data.name_labels = []
                        data.num_classes = []
                        for i_l in range(len(patient_label)):
                            p_l = [i[i_l] for i in [i[2] for i in IndexAndClass]]
                            p_l_aux = sorted(list(set(p_l)))
                            if 'None' in p_l_aux:
                                p_l_aux.remove('None')
                            data.name_labels.append(p_l_aux)
                            data.num_classes.append(len(p_l_aux))
                        
                        # Generate global mean and std per feature
                        if GraphIndex==0:
                            x_ALL = x
                            MeanList = [x.mean((0))*x.shape[0]]
                            VarList = [x.var((0))*x.shape[0]]
                            NumNodesList = [x.shape[0]]
                            data.mean_STD = [MeanList[-1]/sum(NumNodesList), (VarList[-1]/sum(NumNodesList)).sqrt()]
                            data.Percentile = np.percentile(x, 99, axis=0)
                            data.Percentilee = np.percentile(x, 97, axis=0)
                            data.Percentileee = np.percentile(x, 95, axis=0)
                            
                        else:                               
                            MeanList.append(x.mean((0))*x.shape[0])
                            VarList.append(x.var((0))*x.shape[0])
                            NumNodesList.append(x.shape[0])
                            data.mean_STD = [MeanList[-1]/sum(NumNodesList), (VarList[-1]/sum(NumNodesList)).sqrt()]
                            data.Percentile = np.percentile(x, 99, axis=0)
                            data.Percentilee = np.percentile(x, 97, axis=0)
                            data.Percentileee = np.percentile(x, 95, axis=0)
                        # Save each graph in each structure
                        torch.save(data, osp.join(self.processed_dir_graphs, 'data_{}_{}.pt'.format(GraphIndex,SubImageIndex)))
                        SubImageIndex += 1
                GraphIndex += 1
 
            break
    def saveInductiveClusters(self,InductiveClusters, fileIndex, subImId, batch_id,args):
        makedirs(osp.join(self.processed_dir,'ProcessedImages'))
        if args['Phenotypes']:
            np.save(self.processed_dir_cell_types+'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(fileIndex, subImId, InductiveClusters[0].shape[-1]),torch.Tensor.cpu(InductiveClusters[0][batch_id,:]).detach().numpy())     
            np.save(self.processed_dir_cell_types+'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(fileIndex, subImId, InductiveClusters[1].shape[-1]),torch.Tensor.cpu(InductiveClusters[1][batch_id,:]).detach().numpy())     
        else:
            np.save(self.processed_dir_cell_types+'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(fileIndex, subImId, InductiveClusters[0].shape[-1]),torch.Tensor.cpu(InductiveClusters[0][batch_id,:]).detach().numpy())     
    
    def save_cluster_and_attention(self,batch_id, fileIndex,save_Inductivecluster_presence, cluster_assignment, attentionVect,cluster_interactions):        
        for ClustLvl in save_Inductivecluster_presence:
            np.save(self.processed_dir_cell_types+'cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(fileIndex, ClustLvl.shape[-1]),torch.Tensor.cpu(ClustLvl[batch_id,:]).detach().numpy())     
        for ClustLvl in cluster_assignment:
            np.save(self.processed_dir_cell_types+'cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(fileIndex, ClustLvl.shape[-1]),torch.Tensor.cpu(ClustLvl[batch_id,:]).detach().numpy())     
        for ClustLvl in attentionVect:
            np.save(self.processed_dir_cell_types+'attentionVect_Index_{}_ClustLvl_{}.npy'.format(fileIndex,ClustLvl.shape[-1]),torch.Tensor.cpu(ClustLvl[batch_id,:]).detach().numpy())    
        # for ClustLvl in cluster_interactions:
        #     np.save(osp.join(join,'cluster_assignmnt_interaction_Index_{}_ClustLvl_{}.npy'.format(fileIndex,ClustLvl.shape[-1])),torch.Tensor.cpu(ClustLvl[batch_id,:]).detach().numpy())    
    
    def clusterAtt(self, clusters, IndexAndClass, num_classes):
        # Initialize heatmap Cluster Attention 
        if len(clusters)==1:            
            heatmapAttPresence = {str(clusters[0]):[]}
        elif len(clusters)==2:            
            heatmapAttPresence = {str(clusters[0]):[], str(clusters[1]):[]}
        elif len(clusters)==3:            
            heatmapAttPresence = {str(clusters[0]):[], str(clusters[1]):[], str(clusters[2]):[]}
            
        # Cluster Attention per sample 
        linkage_ATT=[]
        for idx, ClusterLevel in enumerate(clusters):
            heatmapClusterAttention = np.zeros([len(IndexAndClass),ClusterLevel])
            labels=[]
            # Extract Attention Vector for each slide
            for idc, idxclster in enumerate(IndexAndClass):                
                attentionVect = np.load(self.processed_dir_cell_types+'attentionVect_Index_{}_ClustLvl_{}.npy'.format(idxclster[1], ClusterLevel))
                # data = self.get(idc,self.())
                heatmapClusterAttention[idc,:] = attentionVect
                if 'Images-MouseBreast' in self.processed_dir:
                    labels.append(int(idxclster[2]))
                elif 'Endometrial_LowGrade' in self.processed_dir: 
                    labels.append(int(idxclster[2]))
                elif 'Synthetic' in self.processed_dir: 
                    labels.append(int(idxclster[2]))
                elif 'Images-Cytof52Breast' in self.root:   
                    labels.append(int(idxclster[2]))
                elif 'ZuriBasel' in self.root:   
                    labels.append(int(idxclster[2]))
            
            # Assign colors to slide-labels
            labels_pal = sns.cubehelix_palette(num_classes, light=.9, dark=.1, reverse=True, start=0, rot=-2)
            labels_lut = dict(zip(map(str, list(set(labels))), labels_pal))        
            labels_colors = [labels_lut[str(k)] for k in labels]
            
            # Create ClusterMap and save its
            heatmapClusterAttention_Fig = sns.clustermap(heatmapClusterAttention,col_cluster=False, row_colors=labels_colors, linewidths=0, cmap="vlag")            
            # Save linkage cluster every heatmap on the same way
            linkage_ATT.append(heatmapClusterAttention_Fig.dendrogram_row.linkage)
            for label in list(set(labels)):
                heatmapClusterAttention_Fig.ax_col_dendrogram.bar(0, 0, color=labels_lut[str(label)], label=label, linewidth=0)        
            # Save 
            heatmapAttPresence[str(ClusterLevel)] = heatmapClusterAttention
            heatmapClusterAttention_Fig.ax_col_dendrogram.legend(title='Class', loc="center", ncol=5, bbox_to_anchor=(0.47, 0.8), bbox_transform=plt.gcf().transFigure)
            heatmapClusterAttention_Fig.savefig(osp.join(self.processed_dir,'Cluster','heatmap_ClusterAttention_nClusters_{}.png'.format(ClusterLevel)))         # Cluster Presence per sample & phenotype presence for each cluster
        return linkage_ATT, heatmapAttPresence, labels

    def clusterPresence(self, clusters, IndexAndClass, num_classes,ClusterThreshold, isTraining):
        ''' Obtain cluster map from each slide.'''
       
        heatmapClusterPresence, labels = obtain_celltype_abundance_per_patient(self, IndexAndClass, clusters, ClusterThreshold)
        
        if not isTraining:
            save_celltype_abundance_per_patient(self,IndexAndClass, clusters, heatmapClusterPresence, labels)

        return heatmapClusterPresence, labels


    def clusterALL(self, clusters, IndexAndClass, num_classes, heatmapAttPresence, heatmapClusterPresence,labels):        
        # Create figure
        # Save color information for next Cluster Level          
        labels_pal = sns.cubehelix_palette(num_classes, light=.9, dark=.1, reverse=True, start=0, rot=-2)
        labels_lut = dict(zip(map(str, list(set(labels[str(clusters[0])]))), labels_pal))        
        labels_colors = [labels_lut[str(k)] for k in labels[str(clusters[0])]]
        # Create ClusterMap
        if len(clusters)==1:
            heatmapClusterPresence_Fig = sns.clustermap(np.concatenate((heatmapClusterPresence[str(clusters[0])],heatmapAttPresence[str(clusters[0])]),
            axis=1),col_cluster=False, row_colors=labels_colors, linewidths=0, cmap="vlag")      
        elif len(clusters)==2:
            heatmapClusterPresence_Fig = sns.clustermap(np.concatenate((heatmapClusterPresence[str(clusters[0])],heatmapAttPresence[str(clusters[0])],
            heatmapClusterPresence[str(clusters[1])],heatmapAttPresence[str(clusters[1])]),axis=1),col_cluster=False,
            row_colors=labels_colors, linewidths=0, cmap="vlag")            
        elif len(clusters)==3:
            heatmapClusterPresence_Fig = sns.clustermap(np.concatenate((heatmapClusterPresence[str(clusters[0])],heatmapAttPresence[str(clusters[0])],
            heatmapClusterPresence[str(clusters[1])],heatmapAttPresence[str(clusters[1])],heatmapClusterPresence[str(clusters[2])],
            heatmapAttPresence[str(clusters[2])]),axis=1),col_cluster=False, row_colors=labels_colors, linewidths=0, cmap="vlag")  
        for label in list(set(labels[str(clusters[0])])):
            heatmapClusterPresence_Fig.ax_col_dendrogram.bar(0, 0, color=labels_lut[str(label)], label=label, linewidth=0)
        heatmapClusterPresence_Fig.ax_col_dendrogram.legend(title='Class', loc="center", ncol=5, bbox_to_anchor=(0.47, 0.8), bbox_transform=plt.gcf().transFigure)
        heatmapClusterPresence_Fig.savefig(osp.join(self.processed_dir,'Cluster','heatmap_ClusterALL.png')) 

    def FindDifferencesBetweenGroups(self, heatmapALL, labels, IndexAndClass):        
        pvalueThreshold=0.0001
        statisticalTest = [] # List of columns with statistical difference. 1.p-value, 2. Groups Number, Cluster step, column of the heatmap.
        for heatmap in heatmapALL: # Iterate each heatmap from the cluster's hierarchy
            if max(labels[heatmap])==1:
                groups= {'0':[], '1':[]}
            elif max(labels[heatmap])==2:
                groups= {'0':[], '1':[],'2':[]}
            for idxColumn, column in enumerate(np.transpose(heatmapALL[heatmap])): # Iterate each feature from the patients's matrix.                
                for pat, _ in enumerate(heatmapALL[heatmap]): # Extract each patient index.
                    groups[str(labels[heatmap][pat])].append(column[pat])
                if max(labels[heatmap])==1:
                    stats.kruskal(groups['0'], groups['1'])
                if max(labels[heatmap])==2:                    
                    if (any([i>0 for i in groups['0']]) or any([i>0 for i in groups['1']]) or any([i>0 for i in groups['2']])) and any([i!=1 for i in groups['0']]): # If all values are zero, go to the next...
                        # Nothing = statisticalTest.append([stats.kruskal(groups['0'], groups['1'])[1],['0','1'], heatmap,idxColumn]) if stats.kruskal(groups['0'], groups['1'])[1]<pvalueThreshold else 0
                        # Nothing = statisticalTest.append([stats.kruskal(groups['0'], groups['2'])[1],['0','2'], heatmap,idxColumn]) if stats.kruskal(groups['0'], groups['2'])[1]<pvalueThreshold else 0
                        # Nothing = statisticalTest.append([stats.kruskal(groups['1'], groups['2'])[1],['1','2'], heatmap,idxColumn]) if stats.kruskal(groups['1'], groups['2'])[1]<pvalueThreshold else 0
                        Nothing = statisticalTest.append([stats.kruskal(groups['0'],groups['1'], groups['2'])[1],['0','1','2'], heatmap,idxColumn]) if stats.kruskal(groups['0'],groups['1'], groups['2'])[1]<pvalueThreshold else 0
        return statisticalTest
    
    # def FindMultivariateDifferencesBetweenGroups(self, heatmapALL, labels, IndexAndClass):        
    #     pvalueThreshold=0.001
    #     statisticalTest = [] # List of columns with statistical difference. 1.p-value, 2.Cluster step, 3.column of the heatmap.
        
    #     # Aggregate each heatmap from the cluster's hierarchy
    #     for heatmap in heatmapALL:
            
    #         # Create Model to find the parameters that most define the disease.
    #         model = smapi.OLS(np.array(labels[[heatmap for heatmap in heatmapALL][-1]]), heatmapALL[heatmap])
    #         results = model.fit()
            
    #         # F-Test of the Model.
    #         FTest=results.f_test(np.identity(results.params.shape[0]))
            
    #         # Obtain most significant clusters
    #         SignificantClusters = list(np.where(results.pvalues<pvalueThreshold)[0])
    #         for sC in SignificantClusters:
    #             statisticalTest.append([results.pvalues[sC],heatmap,sC])
    #     return statisticalTest

    def open_Raw_Image(self, idxclster,reverseIndx):
        Channels, Marker_Names = utilz.load_channels(self.root+'Raw_Data/')

        imList = []
        # Open Original Image
        if 'Endometrial_LowGrade' in self.root: # and osp.isfile(self.root+'Raw_Data/Experiment_Information/Image_Labels.xlsx')  
            # Obtain Labels from excel
            patient_to_image_excel = pd.read_excel(self.root+'Raw_Data/Experiment_Information/Patient_to_Image.xlsx')  

            # Find the actual patient
            image_indices = patient_to_image_excel['Image_Name'][patient_to_image_excel['Subject_Name'].isin([idxclster[0]])]            
            
            # Iterate through images corresponding to this patient
            for numImages, im_name in enumerate(list(image_indices)):
                
                # Load image 
                if im_name.split('.')[-1]=='tiff' or im_name.split('.')[-1]=='tif':  
                    if osp.isfile(self.root+'/Raw_Data/Images/'+im_name):
                        im = tifffile.imread(self.root+'/Raw_Data/Images/'+im_name) 
                    else:
                        continue # Go to next iteration

                # The 3rd dimension should be the channel dimension
                if np.argmin(im.shape)==0:
                    shp = im.shape
                    im = np.moveaxis(im,0,2)                    
                elif np.argmin(im.shape)==1:
                    im = np.reshape(im,(im.shape[0]*im.shape[2],im.shape[1]))     
                
                # Choose user-defined channels
                im = im[:,:,Channels]  
                
                # Append to list of images
                imList.append(im)
            im = np.concatenate(imList,axis=1)            
        elif 'Synthetic' in self.root:  
            
            # Obtain file format
            file_format = os.listdir(self.root+'Raw_Data/Images')[0].split('.')[-1]

            # Take 6 Channels
            im = np.load(self.root+'Raw_Data/Images/'+idxclster[0]+'.'+file_format)   
                                                                                            
            # for i in range(im.shape[2]):
            #     im[:,:,i] = im[:,:,i]/im[:,:,i].max()

        elif 'Endometrial_POLE' in self.root:  
            # Take 7 Channels
            im = tifffile.imread(self.root+'/Raw_Data/Images/'+idxclster[0]+'.tif')   
            im = np.moveaxis(im,0,2) 
            im = im[:,:,Channels]  
            # for i in range(7):
                # im[:,:,i], _ = subtract_background_rolling_ball(im[:,:,i], 50, light_background=False,use_paraboloid=False, do_presmooth=False)
                # im[:,:,i] = im[:,:,i]/im[:,:,i].max()
        elif 'Cytof52Breast' in self.root:  
            # Take 7 first Channels
            dirs = os.listdir(self.root+'/Raw/'+str(idxclster[0]))
            im = np.zeros((tifffile.imread(self.root+'/Raw/'+str(idxclster[0])+'/'+dirs[0]).shape[0],tifffile.imread(self.root+'/Raw/'+str(idxclster[0])+'/'+dirs[0]).shape[1],7))
            for i in range(7):
                im[:,:,i] = tifffile.imread(self.root+'/Raw/'+str(idxclster[0])+'/'+dirs[i])                            
            for i in range(6):
                im[:,:,i] = im[:,:,i]/im[:,:,i].max()
        elif 'Lung' in self.root:  
            # Take 7 Channels
            im = tifffile.imread(self.root+'/Raw_Data/Images/'+idxclster[0]+'.tif')                                     
            im = np.moveaxis(im,0,2) 
            im = im[:,:,Channels]  
        elif 'ZuriBasel' in self.root:
            # Take 39 Channels
            patient_to_image_excel = pd.read_excel(self.root+'Raw_Data/Experiment_Information/Patient_to_Image.xlsx')     
            # Find the actual patient
            image_indices = patient_to_image_excel['Image_Name'][patient_to_image_excel['Subject_Name'].isin([idxclster[0]])]                  
            # Iterate through images corresponding to this patient
            for numImages, im_name in enumerate(list(image_indices)):                
                # Load image 
                if im_name.split('.')[-1]=='tiff' or im_name.split('.')[-1]=='tif':  
                    if osp.isfile(self.root+'/Raw_Data/Images/'+im_name):
                        im = tifffile.imread(self.root+'/Raw_Data/Images/'+im_name) 
                    else:
                        continue # Go to next iteration

                # The 3rd dimension should be the channel dimension
                if np.argmin(im.shape)==0:
                    shp = im.shape
                    im = np.moveaxis(im,0,2)                    
                elif np.argmin(im.shape)==1:
                    im = np.reshape(im,(im.shape[0]*im.shape[2],im.shape[1]))     
                
                # Choose user-defined channels
                im = im[:,:,Channels]  
                
                # Append to list of images
                imList.append(im)

            # Join image information in a mosaic
            rows = 0
            cols = 0
            for iml in imList:
                rows += iml.shape[0]
                cols += iml.shape[1]                    
            im = np.zeros((rows,cols,len(Channels)))
            max_col = 0
            for iml in imList:
                im[:iml.shape[0],max_col:max_col+iml.shape[1],:] = iml      
        else:
            # Load Image in its own format.
            if os.listdir(self.root+'Raw_Data/Images/')[-1].split('.')[-1]=='tiff':
                image = tifffile.imread(self.root+'Raw_Data/Images/'+idxclster[0]+'.tiff')                 
            elif os.listdir(self.root+'Raw_Data/Images/')[-1].split('.')[-1]=='tif':
                image = tifffile.imread(self.root+'Raw_Data/Images/'+idxclster[0]+'.tif') 
            elif os.listdir(self.root+'Raw_Data/Images/')[-1].split('.')[-1]=='npy':
                image = np.load(self.root+'Raw_Data/Images/'+idxclster[0]+'.npy')
            if len(image.shape)==3:
                # The 3rd dimension should be the channel dimension
                if np.argmin(image.shape)==0:
                    image = np.moveaxis(image,0,2) 
                elif np.argmin(image.shape)==1:
                    image = np.moveaxis(image,1,2)            
            # Eliminate unwanted channels
            if len(image.shape)==3:
                im = image[:,:,Channels]
            if len(image.shape)==4:
                im = image[Channels,:,:,:]            
            # Append to list of images
            imList.append(im)
        return im, imList

    def nPlex2RGB(self, im):
        if ('Endometrial' in self.root) or ('Cytof52Breast' in self.root) or ('ZuriBasel' in self.root):
            imRGB = np.zeros((im.shape[0],im.shape[1],3))                        
            # im = im/im.max((0,1),keepdims=True)
            imRGB[:,:,2] += im[:,:,0] # DAPI is blue
            imRGB[:,:,0] += im[:,:,1] # 2ndcolr is red
            imRGB[:,:,0] += im[:,:,2]*1 # 3rdcolr is yellow
            imRGB[:,:,1] += im[:,:,2]*1 # 3rdcolr is yellow
            imRGB[:,:,0] += im[:,:,3]*1 # 4thcolr is Orange
            imRGB[:,:,1] += im[:,:,3]*0.5 # 4thcolr is Orange
            imRGB[:,:,0] += im[:,:,4]*1 # 5thcolr is Magenta
            imRGB[:,:,2] += im[:,:,4]*1 # 5thcolr is Magenta                     
            imRGB[:,:,1] += im[:,:,5] # 6thcolr is Green      
            imRGB[:,:,1] += im[:,:,6]*1 # 7thcolr is Cyan
            imRGB[:,:,2] += im[:,:,6]*1 # 7thcolr is Cyan 
            for i in range(3):
                imRGB[:,:,i] = imRGB[:,:,i]/imRGB[:,:,i].max()
        elif 'Synthetic' in self.root:
            imRGB = np.zeros((im.shape[0],im.shape[1],3))                        
            im = im -im.min()
            imRGB[:,:,2] += im[:,:,0] # DAPI is blue
            imRGB[:,:,0] += im[:,:,1] # 2ndcolr is red
            imRGB[:,:,0] += im[:,:,2]*1 # 3rdcolr is yellow
            imRGB[:,:,1] += im[:,:,2]*1 # 3rdcolr is yellow
            imRGB[:,:,0] += im[:,:,3]*1 # 4thcolr is Orange
            imRGB[:,:,1] += im[:,:,3]*0.5 # 4thcolr is Orange
            imRGB[:,:,0] += im[:,:,4]*1 # 5thcolr is Magenta
            imRGB[:,:,2] += im[:,:,4]*1 # 5thcolr is Magenta                     
            imRGB[:,:,1] += im[:,:,5] # 6thcolr is Green      
            for i in range(3):
                imRGB[:,:,i] = imRGB[:,:,i]/imRGB[:,:,i].max()
        elif 'AlfonsoCalvo' in self.root:
            imRGB = np.zeros((im.shape[0],im.shape[1],3))                        
            imRGB[:,:,2] += im[:,:,0] # DAPI is blue
            imRGB[:,:,0] += im[:,:,2] # CD8 is red                                                                
            imRGB[:,:,0] += im[:,:,3]*0.5 # 5thcolr is Magenta
            imRGB[:,:,2] += im[:,:,3]*0.5 # 5thcolr is Magenta                     
            imRGB[:,:,1] += im[:,:,1] # CD4 is Green      
            for i in range(3):
                imRGB[:,:,i] = imRGB[:,:,i]/imRGB[:,:,i].max()
        return imRGB

    

    def ObtainIntersectInSynthetic(self, statisticalTests, clusters, IndexAndClass, num_classes, attentionLayer):
        # statisticalTest: List of columns with statistical difference. 1.p-value, 2. Groups Number, 3.Cluster step, 4. column of the heatmap        
        statisticalTests = sorted(statisticalTests, key=lambda k: k[0]) 
        stsTest=statisticalTests[0]
        IntersecIndex=[]
        for countSts, stsTest in enumerate(statisticalTests):
            IntersecIndex.append([])
            # Create directory for this set of images.
            thisFolder = self.processed_dir+'/ProcessedImages/BlueIsCluster_GreenIsGroundTruth_ClusterLevel{}_Cluster{}'.format(stsTest[3],stsTest[2])
            if not os.path.exists(thisFolder):
                os.makedirs(thisFolder)
            # For each image...
            for count, idxclster in enumerate(IndexAndClass): 
                # Apply mask to patch
                for patchIDX in range(self.findLastIndex(idxclster[1])+1):                       
                    # Pixel-to-cluster Matrix
                    clust0 = np.load(osp.join(self.processed_dir,'Cluster','cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(idxclster[1], patchIDX, clusters[0])))                    
                    if len(clusters)>1:
                        clust1 = np.load(osp.join(self.processed_dir,'Cluster','cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(idxclster[1], clusters[1])))                    
                        clust1 = np.matmul(clust0,clust1) 
                    if len(clusters)>2:
                        clust2 = np.load(osp.join(self.processed_dir,'Cluster','cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(idxclster[1], clusters[2])))                    
                        clust2 = np.matmul(clust1,clust2)
                    clust = clust0 
                    clust = clust1 if len(clusters)==2 else clust
                    clust = clust2 if len(clusters)==3 else clust                
                    
                    # Assign significant cluster as 1 to Superpixel Image
                    if 'Synthetic' in self.root:
                        Patch_im = np.load(osp.join(self.root,'Original','{}.npy'.format('Labels'+idxclster[0][11:])))-1
                    CLST_suprpxlVal = np.zeros((Patch_im.shape[0],Patch_im.shape[1]),dtype=int)
                    cell_type_top1 = np.apply_along_axis(np.argmax, axis=1, arr=clust)
                    if 'Synthetic' in self.root: 
                        cell_type_top1 = stsTest[3]==cell_type_top1
                    Patch_im=cell_type_top1[Patch_im]
                    # for r in range(Patch_im.shape[0]):
                    #     for c in range(Patch_im.shape[1]):                            
                    #         CLST_suprpxlVal[r,c] = cell_type_top1[Patch_im[r,c]]
                    
                    # Load Ground-Truth
                    if 'SyntheticV2' in self.root: # fenotipo 4 y 5 de la region 2
                        Ground_Truth = np.load(self.root+'/Original/Ground_Truth'+idxclster[0][11:]+'.npy')==2
                    elif 'SyntheticV1' in self.root: # fenotipo 6 de la region 3
                        Ground_Truth = np.load(self.root+'/Original/Ground_Truth'+idxclster[0][11:]+'.npy')==3
                    
                    # Calculate Intersection Index
                    intersection = np.logical_and(Patch_im, Ground_Truth)                    
                    IntersecIndex[countSts].append(intersection.sum() / float(Patch_im.sum()))

                    # Save GT and Image
                    RGBImage = np.zeros((Ground_Truth.shape[0],Ground_Truth.shape[1],3))
                    RGBImage[:,:,1] = Ground_Truth
                    RGBImage[:,:,2] = Patch_im
                    plt.imsave(thisFolder+'/IntersectIdx{}_Slide{}_Patch{}.png'.format(IntersecIndex[countSts][-1],idxclster[1],patchIDX), RGBImage)                

        return IntersecIndex
    
    def ObtainClustATT(self,GraphIndex,ClusterLevel,attentionLayer,cluster_assignment_attn,idx): 
        ''' From some indices obtain the cluster map'''       
        # Obtain Clustering
        cluster_assignment = np.load(osp.join(self.processed_dir,'Cluster','cluster_assignment_Index_{}_ClustLvl_{}.npy'.format(GraphIndex, ClusterLevel)))
        # Obtain attention            
        attntn = np.load(osp.join(self.processed_dir,'Cluster','attentionVect_Index_{}_ClustLvl_{}.npy'.format(GraphIndex, ClusterLevel))) if attentionLayer else np.ones(cluster_assignment.shape[-1])
        # Obtain 2nd level clusterpresence from attention, and from the 2nd level clustering
        if idx>0:                                    
            cluster_assignment = np.matmul(cluster_assignment_attn,cluster_assignment)
        cluster_assignment_attn = (attntn*cluster_assignment)/(attntn*cluster_assignment).sum()
        cluster_assignment_prev = copy.deepcopy(cluster_assignment_attn)
        return cluster_assignment,cluster_assignment_attn, attntn
    
    
    def HeatmapMarkerExpression(self,clusters,IndexAndClass,num_classes,ClusteringThreshold,cell_type):
        
        if 'Superpixel' in self.raw_dir:
            # Obtain clusters of the first label. We want to obtain the phenotype!
            ClusterLevel=clusters
            with open(self.root+'/OriginalSuperpixel/Superpixels_Names.txt', 'r') as csvfile:
                Superpixels_Names=[]
                read = csv.reader(csvfile, delimiter=',')
                for row in read:
                    Superpixels_Names.append(row[0])
        elif ('GBM' in self.raw_dir) or ('KIRC' in self.raw_dir):
            # Obtain clusters of the first label. We want to obtain the phenotype!
            ClusterLevel=clusters
            with open(self.root+'/Raw_Data/Experiment_Information/CellSegmentationFeatures.txt', 'r') as csvfile:
                Superpixels_Names=[]
                read = csv.reader(csvfile, delimiter=',')
                for row in read:                    
                    Superpixels_Names.append(row[0])
        else:
            # Obtain clusters of the first label. We want to obtain the phenotype!
            ClusterLevel=clusters
            with open(self.root+'/Raw_Data/Experiment_Information/Channels.txt', 'r') as csvfile:
                Superpixels_Names=[]
                read = csv.reader(csvfile, delimiter=',')
                for row in read:                    
                    Superpixels_Names.append(row[0])

        heatmapMarkerExpression = np.zeros((ClusterLevel,len([sn for sn in Superpixels_Names if sn!='None'])))
        NumberOfNodesInEachCluster = np.zeros((ClusterLevel))
        # Obtain clusters per Slide
        for count, idxclster in enumerate(IndexAndClass):       
            
            # Load Cluster Assignment for a specific image
            cluster_assignment = np.load(osp.join(self.processed_dir_cell_types,'cluster_assignmentPerPatch_Index_{}_{}_ClustLvl_{}.npy'.format(idxclster[1],0, ClusterLevel)))            
                        
            # Obtain Superpixel Image
            if 'Endometrial' in self.root:
                suprpxlFeat = np.load(osp.join(self.raw_dir,'{}.npy'.format(idxclster[0])))
                suprpxlFeat = suprpxlFeat[:,2:2+len(Superpixels_Names)]       
            elif 'Synthetic' in self.root:
                if 'Superpixel' in self.raw_dir:
                    suprpxlFeat = np.loadtxt(osp.join(self.root,'OriginalSuperpixel','{}.txt'.format(idxclster[0])),delimiter=',')
                elif 'SuperPatch' in self.raw_dir:
                    suprpxlFeat = np.load(osp.join(self.root,'OriginalSuperPatch','{}.npy'.format(idxclster[0])))
                    suprpxlFeat = suprpxlFeat[:,2:(suprpxlFeat.shape[1]-self.SuperPatchEmbedding)]                    
            elif 'SuperPatch' in self.raw_dir:
                suprpxlFeat = np.load(osp.join(self.root,'OriginalSuperPatch','{}.npy'.format(idxclster[0])))
                suprpxlFeat = suprpxlFeat[:,2:(suprpxlFeat.shape[1]-self.SuperPatchEmbedding)] 
            else:
                suprpxlFeat = np.load(osp.join(self.raw_dir,'{}.npy'.format(idxclster[0])))
                suprpxlFeat = suprpxlFeat[:,2:]
                        
            # Save value to Marker Expression Heatmap
            # cluster_assignment[cluster_assignment<np.percentile(cluster_assignment.sum(-1), ClusteringThreshold)]=1e-15           
            for clustI in range(cluster_assignment.shape[1]):    
                logical_values = (cluster_assignment.argmax(-1)==clustI)[:suprpxlFeat.shape[0]]                                           
                SuperpixelClustI = suprpxlFeat[(logical_values)[:suprpxlFeat.shape[0]],:] 
                if SuperpixelClustI.size > 0: 
                    NumberOfNodesInEachCluster[clustI] += np.array(logical_values[:suprpxlFeat.shape[0]]).sum()             
                    heatmapMarkerExpression[clustI,:] += SuperpixelClustI.mean(-2)                
            
        # Normalize HeatMarkerExpression with respect to the number of cells present there        
        heatmapMarkerExpression = np.nan_to_num(heatmapMarkerExpression)
        # Normalize Z-Score phenotypes to maximize differences between clusters.
        heatmapMarkerExpression = stats.zscore(heatmapMarkerExpression,axis=0)        

        # Extract labels        
        labels = [i[2] for i in IndexAndClass]

        if cell_type=='Neighborhoods':
            celltypes_names = ['N'+str(i+1) for i in range(heatmapMarkerExpression.shape[0])]
        else:
            celltypes_names = ['P'+str(i+1) for i in range(heatmapMarkerExpression.shape[0])]

            
        # Generate HeatMarkerExpression Map            
        Colormap=cm.jet_r(range(0,255,int(255/heatmapMarkerExpression.shape[0])))[:,:3]

        # Create ClusterMap
        plt.close()
        plt.figure()
        heatmapMarkerExpression_Fig = sns.clustermap(heatmapMarkerExpression, vmin=-2, vmax=2,col_cluster=False, row_cluster=False, row_colors=Colormap,xticklabels=Superpixels_Names,yticklabels=celltypes_names, linewidths=0.5, cmap="Spectral_r")            
        heatmapMarkerExpression_Fig.savefig(osp.join(self.bioInsights_dir_cell_types,cell_type,'heatmap_MarkerExpression_nClusters_{}_iter_{}_Thr{}.png'.format(ClusterLevel,self.args['epochs'],ClusteringThreshold)),dpi=300) 

        # Generate BarPlot OF presence of each Cluster in Our Experiment
        fig=plt.figure()
        # reordered_ind = heatmapMarkerExpression_Fig.dendrogram_row.reordered_ind
        BarPlotPresenceOfPhenotypes = sns.barplot(x=np.array(list(range(len(NumberOfNodesInEachCluster)))),y=NumberOfNodesInEachCluster,palette='jet_r')#,hue=np.array(reordered_ind)*int(255/heatmapMarkerExpression.shape[0]),)        
        fff = BarPlotPresenceOfPhenotypes.get_figure()       
        fff.savefig(osp.join(self.bioInsights_dir_cell_types,cell_type,'Barplot_MarkerExpression_nClusters_{}_iter_{}.png'.format(ClusterLevel,self.args['epochs'])))
    
    def RGBtoHex(self, vals, rgbtype=1):
        """Converts RGB values in a variety of formats to Hex values.

            @param  vals     An RGB/RGBA tuple
            @param  rgbtype  Valid valus are:
                                1 - Inputs are in the range 0 to 1
                                256 - Inputs are in the range 0 to 255

            @return A hex string in the form '#RRGGBB' or '#RRGGBBAA'
        """

        if len(vals)!=3 and len(vals)!=4:
            raise Exception("RGB or RGBA inputs to RGBtoHex must have three or four elements!")
        if rgbtype!=1 and rgbtype!=256:
            raise Exception("rgbtype must be 1 or 256!")

        #Convert from 0-1 RGB/RGBA to 0-255 RGB/RGBA
        if rgbtype==1:
            vals = [255*x for x in vals]

        #Ensure values are rounded integers, convert to hex, and concatenate
        return '#' + ''.join(['{:02X}'.format(int(round(x))) for x in vals])

    def Best_and_Worst(self,statisticalTest,num_classes,unrestrictedLoss,IndexAndClass):
        if len(statisticalTest)>0:
            # Obtain Best and Worst Case
            if num_classes==2:
                IndexAndClass_Best_and_Worst = [[],[],[],[]]
            elif num_classes==3:
                IndexAndClass_Best_and_Worst = [[],[],[],[],[],[]]
            
            # Choose the sample that best and worst fits the model.s
            for c in range(num_classes):
                maxVal = unrestrictedLoss.mean()
                minVal = unrestrictedLoss.mean()
                for n, ind in enumerate(IndexAndClass):
                    if ind[2]==c:
                        if unrestrictedLoss[n]<minVal:
                            minVal=unrestrictedLoss[n]
                            IndexAndClass_Best_and_Worst[c*2]= ind # Best Sample in class
                        if unrestrictedLoss[n]>maxVal:
                            maxVal=unrestrictedLoss[n]
                            IndexAndClass_Best_and_Worst[1+c*2]= ind  # Worst Sample in class
        return IndexAndClass_Best_and_Worst
            
    def ObtainPhenoExamples(self,IndexAndClass,clusters):
        '''
        Show information for each phenotype and neighborhood.
        '''
        
        if len([i for i in os.listdir(self.bioInsights_dir_cell_types_Neigh) if 'Interactivity' in i])>0 and False:
            return

        CropConfPheno, CropConfTissueComm = select_patches_from_cohort(self,IndexAndClass,clusters)                  
        
        Channels, Marker_Names = utilz.load_channels(self.root+'Raw_Data/')

        # Extract topk patches and save them
        heatmapMarkerExpression, heatmap_MarkerColocalization = extract_topk_patches_from_cohort(self, CropConfPheno, [Marker_Names[c] for c in Channels],'Phenotypes')
        save_heatMapMarker_and_barplot(self, heatmapMarkerExpression, heatmap_MarkerColocalization,CropConfPheno,[Marker_Names[c] for c in Channels],'Phenotypes')

        # Extract topk patches and save them
        heatmapMarkerExpression, heatmap_MarkerColocalization = extract_topk_patches_from_cohort(self, CropConfTissueComm, [Marker_Names[c] for c in Channels],'Neighborhoods')
        save_heatMapMarker_and_barplot(self, heatmapMarkerExpression, heatmap_MarkerColocalization,CropConfTissueComm,[Marker_Names[c] for c in Channels],'Neighborhoods')

        # Neighborhoods to Phenotypes
        # obtain_neighborhood_composition(self,CropConfPheno,CropConfTissueComm)

    def visualize_results(self,model, clusters, IndexAndClass, num_classes,mean_STD, args):        
        '''
            Visualize all the results.
        '''

        # Eliminate those patients 
        IndexAndClass = [iac for iac in IndexAndClass if 'None'!=iac[2][0]]
        
        # Obtain Neigh-to-Pheno, and Area-to-Neigh heatmap.
        for ClusteringTrheshold in [0,50,75,90,95]:
            Area_to_Neighborhood_to_Phenotype(self,clusters,IndexAndClass,num_classes,ClusteringTrheshold)

        # Obtain cell-types examples
        self.ObtainPhenoExamples(IndexAndClass,clusters)        
        
        # Cell-types abundances
        heatmapClusterPresence, labels = self.clusterPresence(clusters, IndexAndClass, num_classes,0, isTraining=False)                        
        
        # Differential abundance analysis
        statisticalTest, unrestrictedLoss, Top1PerPatient, patient_Ineach_subgroup, real_class_confidence = differential_abundance_analysis(self, heatmapClusterPresence, labels, IndexAndClass,False)

        # Obtain Segmentation index for Statistical Value..
        if ('Synthetic' in self.root):
            IntersectionIndex = ObtainMultivariateIntersectInSynthetic(self, Top1PerPatient, clusters, IndexAndClass, num_classes, args['isAttentionLayer'],False)

        # Extract visual cues of sinificant clusters                   
        IntersectionIndex = TME_location_in_image(self, patient_Ineach_subgroup, clusters, IndexAndClass,[real_class_confidence[i] for i in patient_Ineach_subgroup['Patient index']],Top1PerPatient, num_classes, args['isAttentionLayer'],[0,50,75,90])        
        
        # Apply clusters and attention to images
        All_TMEs_in_Image(self, clusters, patient_Ineach_subgroup, IndexAndClass, num_classes, args['isAttentionLayer'],0)        
   
    def obtain_intersec_acc(self,model, clusters, IndexAndClass, num_classes,mean_STD, args,TrainingClusterMapEpoch):        

        isTraining = True
                   
        # Obtain patient-CLusterMap and calculate statisticalSignificance between groups.       
        self.TrainingClusterMapEpoch =  TrainingClusterMapEpoch
        heatmapClusterPresence, labels = self.clusterPresence(clusters, IndexAndClass, num_classes, 0, isTraining=True)  
        statisticalTest, unrestrictedLoss, Top1PerPatient, patient_Ineach_subgroup, real_class_confidence = differential_abundance_analysis(self, heatmapClusterPresence, labels, IndexAndClass,True)

        # Obtain Segmentation index for Statistical Value.
        if ('Synthetic' in self.root):
            IntersectionIndex = ObtainMultivariateIntersectInSynthetic(self, Top1PerPatient, clusters, IndexAndClass, num_classes, args['isAttentionLayer'],True)
        else:
            IndexAndClass_Best_and_Worst = Best_and_Worst(statisticalTest,num_classes,unrestrictedLoss)
            IntersectionIndex = self.ObtainMultivariateIntersectInReal(statisticalTest, clusters, self.IndexAndClass_onePerClass, num_classes, args['isAttentionLayer'],0.5)
            IntersectionIndex = self.ObtainMultivariateIntersectInReal(statisticalTest, clusters, self.IndexAndClass_onePerClass, num_classes, args['isAttentionLayer'],0.75)
            IntersectionIndex = self.ObtainMultivariateIntersectInReal(statisticalTest, clusters, self.IndexAndClass_onePerClass, num_classes, args['isAttentionLayer'],0.95)
        return np.array(IntersectionIndex[1]).mean()

def get_BioInsights(path, parameters):
    '''
    Code to calculate and obtain all the statistics from the experiment.
    '''
    # Load the model.
    N = NaroNet.NaroNet.NaroNet(parameters, 'cpu')
    N.epoch = 0    
    N.dataset.args = parameters

    # Visualize results
    N.visualize_results()



