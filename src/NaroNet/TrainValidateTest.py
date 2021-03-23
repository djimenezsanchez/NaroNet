import random as rand
import torch
import numpy as np
import copy
import torch
from NaroNet.utils import utilz
import torch.nn.functional as F
import math
import NaroNet.NaroNet_model.loss as loss_Op



def gatherData(self,index, Indices,subgraphs,training,args):
    if training and args['UnsupContrast']:
        if index+int(self.args['batch_size'])>len(Indices):
            data, subgraphs, SelectedsubImIndx=self.dataset.gatherData_UnsupContrast(self, Indices[index:index+int(self.args['batch_size']/2)], subgraphs,training)            
        else:
            data, subgraphs, SelectedsubImIndx=self.dataset.gatherData_UnsupContrast(self, Indices[index:index+int(self.args['batch_size']/2)],subgraphs,training)
    else:
        if index+self.args['batch_size']>len(Indices):
            data, subgraphs, SelectedsubImIndx=self.dataset.gatherData(self, Indices[index:], subgraphs,training)            
        else:
            data, subgraphs, SelectedsubImIndx=self.dataset.gatherData(self, Indices[index:index+self.args['batch_size']],subgraphs,training)
    return data, subgraphs, SelectedsubImIndx

def inductiveClustering(self,Indices,index,saveClusters,trainClustering,training,labels,optimizer):
    '''
    Method that loads subject's respective graph, and insert it
    in NaroNet to calculate the phenotypes and neihborhoods
        Indices: (list of ints) that specifies the indices of the subjects.
        index: (int) that specifies the index from the subjects should be obtained in this minibatch step.
        saveClusters: (boolean) if True the Clusters are saved, if False nothing happens.
        trainClustering: (boolean) if True NaroNet is trained end-to-end, if false NaroNet' clustering layers are trained in an unsupervised way.
        training: (boolean) if True NaroNet is trained, if False subjects are inferred.
        labels: (list of ints) that specifies the labels of the subjects
        optimizer: (object) with the specified NaroNet optimizer.
    '''
    
    def clustToTissueGraph(SelectedsubImIndx, chosenLabels, saveClusterPresencePheno, s_iter, saveClusterPresenceComm, total_num_nodes,training):
        if self.args['UnsupContrast'] and training:
            for b, subIm in enumerate(SelectedsubImIndx):
                if len(self.data.y)==0:
                    aux2 = []
                    for i in chosenLabels[:len(SelectedsubImIndx)]:
                        aux2.append(i)
                        aux2.append(i)
                    self.data.y = torch.tensor(aux2,dtype=torch.float32).to(device=self.device)
                self.data.x[subIm[1]*2,:,:] += self.dataNOW.x.data[b*2,:,:]*self.dataNOW.num_nodes[b*2]
                self.data.x[1+subIm[1]*2,:,:] += self.dataNOW.x.data[1+b*2,:,:]*self.dataNOW.num_nodes[1+b*2]
                self.data.edge_index[subIm[1]*2,:,:] += self.dataNOW.edge_index.data[b*2,:,:]*self.dataNOW.num_nodes[b*2]
                self.data.edge_index[1+subIm[1]*2,:,:] += self.dataNOW.edge_index.data[1+b*2,:,:]*self.dataNOW.num_nodes[1+b*2]
                total_num_nodes[subIm[1]] += self.dataNOW.num_nodes[b*2]
                total_num_nodes[1+subIm[1]*2] += self.dataNOW.num_nodes[1+b*2]

        else:            
            for b, subIm in enumerate(SelectedsubImIndx):
                if len(self.data.y)==0:
                    self.data.y = torch.tensor(chosenLabels,dtype=torch.float32).to(device=self.device)
                if self.args['Phenotypes']:
                    saveClusterPresencePheno[subIm[1],:] += s_iter[0][b,:self.dataNOW.num_nodes[b],:].sum(-2)
                saveClusterPresenceComm[subIm[1],:] += s_iter[-1][b,:self.dataNOW.num_nodes[b],:].sum(-2)
                self.data.x[subIm[1],:,:] += self.dataNOW.x.data[b,:,:]*self.dataNOW.num_nodes[b]
                self.data.edge_index[subIm[1],:,:] += self.dataNOW.edge_index.data[b,:,:]*self.dataNOW.num_nodes[b]
                total_num_nodes[subIm[1]] += self.dataNOW.num_nodes[b]                
            # Save Clustering of patch from a specific patch.
            if saveClusters:
                for b, subIm in enumerate(SelectedsubImIndx):
                    self.dataset.saveInductiveClusters(s_iter, Indices[index+subIm[1]],subIm[0][0],b, self.args)      
        return saveClusterPresencePheno, saveClusterPresenceComm, total_num_nodes


    def subgraph_init():
        '''
        Method that initializes the subgraphs list.
        Outputs:
            subgraphs: (list of ints) that specifies the number of subgraphs a single sbject is composed by.
            chosenLabels: (list of ints) that specifies the label of each subgraph for each subject.
        '''

        # Initialize the subgraphs list and the chosenLabels
        subgraphs=[] 
        chosenLabels=[]
        
        # Iterate to obtain the number of subgraphs each subject is composed by.
        for IndicesI in Indices[index:min(index+self.args['batch_size'],len(Indices))]:             
            
            # Obtain the labels of this subject.
            chosenLabels.append(labels[IndicesI])
            
            # If True the subject has a single subgraph, if False obtain the number of subgraphs.
            if self.dataset.findLastIndex(IndicesI)==0:
                subgraphs.append([0]) 
            else:
                subgraphs.append(list(range(0,self.dataset.findLastIndex(IndicesI)))) 
        
        return subgraphs, chosenLabels
    
    # Initialize the subgraphs list
    subgraphs, chosenLabels = subgraph_init()

    # Initialize an empty graph
    if self.args['UnsupContrast'] and training:        
        self.data = self.dataset.generateEmptyClusteringGraph(min([len(subgraphs)*2,self.args['batch_size']]),self.args['clusters'][1],self.args['hiddens'])         
    else:
        self.data = self.dataset.generateEmptyClusteringGraph(len(subgraphs),self.args['clusters'][1],self.args['hiddens']) 
    self.data.x = torch.Tensor(self.data.x).to(self.device)
    self.data.edge_index = torch.Tensor(self.data.edge_index).to(self.device)
    
    # Empty value to save the number of nodes in a slide
    total_num_nodes=np.zeros(self.args['batch_size'])
    
    # Initialize the outputs
    ortho_color_total = []
    ortho_total = []
    cell_ent_total = []
    pat_ent_total = []
    MinCUT_total = []
    pearsonCoeffUNSUP_total=[]
    pearsonCoeffSUP_total=[]
    
    # Used to save cluster presence for visualization
    if self.args['UnsupContrast']:
        saveClusterPresencePheno = torch.zeros([min([len(subgraphs)*2,self.args['batch_size']]),int(self.args['clusters1'])])
        saveClusterPresenceComm = torch.zeros([min([len(subgraphs)*2,self.args['batch_size']]),int(self.args['clusters2'])])
    else:
        saveClusterPresencePheno = torch.zeros([len(subgraphs),int(self.args['clusters1'])])
        saveClusterPresenceComm = torch.zeros([len(subgraphs),int(self.args['clusters2'])])

    # Iterate each subject to load
    while any([len(lista)>0 for lista in subgraphs]):
        
        # Initialize the optimizer 
        optimizer.zero_grad()

        # Obtain data from folder.
        self.dataNOW,subgraphs, SelectedsubImIndx = gatherData(self,index,Indices,subgraphs,training,self.args)        
        self.dataNOW = self.dataNOW.to(self.device)
        
        # Cluster and embedding from model.
        self.dataNOW,ortho_color0,ortho_color1,MinCUT0,MinCUT1,ortho0,ortho1,cell_ent_loss0,cell_ent_loss1,pat_ent0, pat_ent1,pearsonCoeffSUP,pearsonCoeffUNSUP,s_iter = self.model(self.dataNOW,self.device,saveClusters,trainClustering,True,index,Indices,labels,self.args)       
        
        # Unsupervised losses
        ortho_color_total.append((ortho_color0.item()+ortho_color1.item())/2) if self.args['orthoColor'] else ortho_color_total.append(0)
        ortho_total.append((ortho0.item()+ortho1.item())/2) if self.args['ortho'] else ortho_total.append(0)
        cell_ent_total.append((cell_ent_loss0.item()+cell_ent_loss1.item())/2) if self.args['min_Cell_entropy'] else cell_ent_total.append(0)
        pat_ent_total.append((pat_ent0.item()+pat_ent1.item())/2) if self.args['Max_Pat_Entropy'] else pat_ent_total.append(0)
        MinCUT_total.append((MinCUT0.item()+MinCUT1.item())/2) if self.args['MinCut'] else MinCUT_total.append(0) 
        pearsonCoeffSUP_total.append(pearsonCoeffSUP.item()) if self.args['pearsonCoeffSUP'] else pearsonCoeffSUP_total.append(0)        
        pearsonCoeffUNSUP_total.append(pearsonCoeffUNSUP.item()) if self.args['pearsonCoeffUNSUP'] else pearsonCoeffUNSUP_total.append(0)         
        
        # tensor loss.
        loss = self.args['orthoColor_Lambda0']*ortho_color0[0] + self.args['orthoColor_Lambda1']*ortho_color1[0]  if self.args['orthoColor'] else 0
        loss += self.args['ortho_Lambda0']*ortho0 + self.args['ortho_Lambda1']*ortho1  if self.args['ortho'] else 0
        if self.args['Max_Pat_Entropy']: 
            loss += self.args['Max_Pat_Entropy_Lambda0']*pat_ent0 + self.args['Max_Pat_Entropy_Lambda1']*pat_ent1
            # print('Pat_ent: Phenotypes:',pat_ent0.item(),'T-Comm:',pat_ent1.item())
        if  self.args['min_Cell_entropy']:
            loss += self.args['min_Cell_entropy_Lambda0']*cell_ent_loss0 + self.args['min_Cell_entropy_Lambda1']*cell_ent_loss1
            # print('cell_ent: Phenotypes:',cell_ent_loss0.item(),'T-Comm:',cell_ent_loss1.item())
        if  self.args['MinCut']:
            loss += self.args['MinCut_Lambda0']*MinCUT0 + self.args['MinCut_Lambda1']*MinCUT1            
        loss += (pearsonCoeffSUP)        
        loss += (pearsonCoeffUNSUP)                       

        # Assign clustering to tissue-graph
        saveClusterPresencePheno, saveClusterPresenceComm, total_num_nodes = clustToTissueGraph(SelectedsubImIndx, chosenLabels, saveClusterPresencePheno, s_iter, saveClusterPresenceComm, total_num_nodes,training)

        if training and not self.args['learnSupvsdClust']: #and any([len(lista)>0 for lista in subgraphs]):
            if sum([len(i)>0 for i in subgraphs])>len(subgraphs)*0.75:
                # dot = make_dot(self.model.S[0].sum().detach)
                # dot.format = 'png'    
                # dot.render('/gpu-data/djsanchez/aaj')
                utilz.apply_loss(self.model,loss,self.optimizer)            
            else:

                utilz.apply_loss(self.model,loss,self.optimizer)            
        elif training and self.args['learnSupvsdClust']:
            # Want to train supervised Clustering
            # total_num_nodes=np.ones(batch_size) # We dont want to normalize the interaction-graph.
            break  

    if not self.args['learnSupvsdClust']:
        self.model.S = [torch.zeros(saveClusterPresencePheno.shape),torch.zeros(saveClusterPresenceComm.shape)]
        for b in range(self.data.x.shape[0]):
            self.data.x[b,:,:] = self.data.x[b,:,:]/total_num_nodes[b]
            self.data.edge_index[b,:,:] = self.data.edge_index[b,:,:]/total_num_nodes[b]
            self.model.S[0][b,:] = saveClusterPresencePheno[b,:]/total_num_nodes[b]
            self.model.S[1][b,:] = saveClusterPresenceComm[b,:]/total_num_nodes[b]                
        self.model.S[0] = self.model.S[0].detach().to(self.device)
        self.model.S[1] = self.model.S[1].detach().to(self.device)
        self.data.x = self.data.x.detach()
        self.data.edge_index = self.data.edge_index.detach()
    return self.data, np.mean(ortho_color_total), np.mean(ortho_total),np.mean(cell_ent_total),np.mean(pat_ent_total), np.mean(MinCUT_total), [saveClusterPresencePheno,saveClusterPresenceComm], loss, np.mean(pearsonCoeffSUP_total), np.mean(pearsonCoeffUNSUP_total)

def train(self,Indices,optimizer,training,trainClustering,saveClusters,labels):
    '''
    We train/test the specified subjects using NaroNet
    Indices: (list of ints) that specifies the indices of the images.
    optimizer: (object) with the specified NaroNet optimizer.
    training: (boolean) if True NaroNet is trained, if False subjects are inferred.
    trainClustering: (boolean) if True NaroNet is trained end-to-end, if false NaroNet' clustering layers are trained in an unsupervised way.
    saveClusters: (boolean) if True the Clusters are saved, if False nothing happens.
    labels: (list of ints) that specifies the labels of the subjects
    '''

    # Initialize the outputs
    correct = 0
    total_loss = 0
    total_ortho_color=0    
    total_ortho=0
    total_MinCUT_loss=0
    total_pearsonCoeffSUP=0
    total_pearsonCoeffUNSUP=0
    total_nearestNeighbor_loss=0
    total_UnsupContrast_loss =0
    total_UnsupContrastAcc = 0
    total_Pat_ent = 0
    total_cross_ent = 0
    total_cell_ent = 0

    # Shuffle Indices if we are training the model
    rand.shuffle(Indices)
    
    # Start minibatch to train/test the model
    for index in range(0,len(Indices),self.args['batch_size']):   

        # index = max(len(Indices)-self.args['batch_size'],0) if abs(index-len(Indices))<self.args['batch_size'] else index
        # print(str(index) + '-' + str(len(Indices)))
        # If True NaroNet is trained, if False subjects are inferred
        if training:  
            
            # Train NaroNet
            self.model.train()

            # Obtain phenotypes and neighborhoods.            
            data, ortho_color_ind, ortho_ind, cell_ent_ind,pat_ent_ind, mincut_ind, save_Inductivecluster_presence, loss_induc, pearsonCoeffSUP_loss, pearsonCoeffUNSUP_loss = inductiveClustering(self,Indices,index,saveClusters,trainClustering,training,labels,optimizer)

            # Obtain neighborhood interactions, and classify the subjects.
            out, ortho_color, MinCUT_loss, ortho, cluster_assignment, cluster_interaction, unsup_loss, attentionVector, pearsonCoeffSUP, pearsonCoeffUNSUP,nearestNeighbor_loss, f_test_loss, UnsupContrast, Pat_ent, Cell_ent = self.model(data,self.device,saveClusters,trainClustering,False,index,Indices,labels,self.args)                        
        
        else:            
            
            # Subjects are inferred. 
            self.model.eval()    

            # Specify that NaroNet does not calculate the graph of gradients.
            with torch.no_grad():            
                
                # Obtain interaction graph from patches
                doClustering=True
                data, ortho_color_ind, ortho_ind, cell_ent_ind,pat_ent_ind, mincut_ind, save_Inductivecluster_presence, loss_induc, pearsonCoeffSUP_loss, pearsonCoeffUNSUP_loss = inductiveClustering(self,Indices,index,saveClusters,trainClustering,training,labels,optimizer)                
                
                # insert interaction graph to obtain a patient-label
                doClustering=False
                out, ortho_color, MinCUT_loss, ortho, cluster_assignment, cluster_interaction, unsup_loss, attentionVector, pearsonCoeffSUP, pearsonCoeffUNSUP,nearestNeighbor_loss, f_test_loss, UnsupContrast, Pat_ent, Cell_ent = self.model(data,self.device,saveClusters,trainClustering,doClustering,index,Indices,labels,self.args)
        
        Cross_entropy_loss, pred_Cross_entropy, PredictedLabels_Cross_entropy = utilz.cross_entropy_loss(training, self.args, out, data.y,self.dataset,self.device,self.model)                
        
        
        if self.args['Lasso_Feat_Selection']:
            Lasso = []
            if len(self.args['experiment_Label'])==1:
                Lasso.append(loss_Op.Lasso_Feat_Selection(self.model.lin1_1,self.model.lin2_1)*self.args['Lasso_Feat_Selection_Lambda0'])            
            if len(self.args['experiment_Label'])==2:
                Lasso.append(loss_Op.Lasso_Feat_Selection(self.model.lin1_2,self.model.lin2_2)*self.args['Lasso_Feat_Selection_Lambda1'])            
            if len(self.args['experiment_Label'])==3:
                Lasso.append(loss_Op.Lasso_Feat_Selection(self.model.lin1_3,self.model.lin2_3)*self.args['Lasso_Feat_Selection_Lambda2'])            
            if len(self.args['experiment_Label'])==4:
                Lasso.append(loss_Op.Lasso_Feat_Selection(self.model.lin1_4,self.model.lin2_4)*self.args['Lasso_Feat_Selection_Lambda3'])            
        else:
            Lasso=[0]
        
        # Learnable learning rate for each loss
        # print('Learning Rate: ortho:',self.model.lr_ortho.data.item())#,' unsup:', self.model.lr_unsup.data.item())

        # Join losses
        loss = utilz.gather_apply_loss(training,loss_induc,Lasso, nearestNeighbor_loss, f_test_loss, UnsupContrast, Cross_entropy_loss, ortho, MinCUT_loss, ortho_color, pearsonCoeffSUP, pearsonCoeffUNSUP, Pat_ent,self.optimizer,self.model,self.args)    
        # self.model.NNx.detach()
        # self.model.NNCPosition.detach()

        # Save predictions and GT for the statistics      
        # if not self.args['ObjectiveCluster']:
        pred, PredictedLabels = pred_Cross_entropy, PredictedLabels_Cross_entropy
        # if self.args['NearestNeighborClassification']:
        #     pred, PredictedLabels = pred_Nearesteighbor, PredictedLabels_NearestNeighbor
        GroundTruthLabels=data.y.cpu().numpy()
        if 'PredictedLabelsAll' in locals():
            for pred_all_i in range(len(PredictedLabelsAll)):
                PredictedLabelsAll[pred_all_i] = np.concatenate((PredictedLabelsAll[pred_all_i],PredictedLabels[pred_all_i]))            
            GroundTruthLabelsAll = np.concatenate((GroundTruthLabelsAll,GroundTruthLabels))
        else:
            PredictedLabelsAll = PredictedLabels
            GroundTruthLabelsAll = GroundTruthLabels        
        for i in range(len(pred)):
            if pred[i].max()>5:
                correct += np.abs(data.y[:,i].cpu().numpy()-pred[i]).sum()        
            else:                
                correct += np.equal(pred[i],data.y[:,i].cpu().numpy()).sum()        

        # Save supervised/unsupervised loss
        total_loss += loss.item() if torch.is_tensor(loss) else 0
        total_ortho_color += ortho_color_ind.item() if self.args['orthoColor'] else 0
        total_ortho += ortho.item() if self.args['ortho'] and len(cluster_assignment)>0 else 0        
        # total_MinCUT_loss += MinCUT_loss.item() if self.args['MinCut'] and len(cluster_assignment)>0 else 0       #  No se suma porque se refiere a la union de Neighborhoods de una manera especifica. Y eso no lo buscamos.
        total_pearsonCoeffSUP += pearsonCoeffSUP.item() if self.args['pearsonCoeffSUP'] and len(cluster_assignment)>0 else 0
        total_pearsonCoeffUNSUP += pearsonCoeffUNSUP.item() if self.args['pearsonCoeffUNSUP'] and len(cluster_assignment)>0 else 0
        total_nearestNeighbor_loss += nearestNeighbor_loss.item() if self.args['NearestNeighborClassification'] and len(cluster_assignment)>0 else 0         
        total_UnsupContrast_loss += UnsupContrast[0].item()+UnsupContrast[2].item()+UnsupContrast[4].item() if self.args['UnsupContrast'] and len(cluster_assignment)>0 and (training) else 0
        total_UnsupContrastAcc += (UnsupContrast[1].item()+UnsupContrast[3].item()+UnsupContrast[5].item())/3 if self.args['UnsupContrast'] and len(cluster_assignment)>0 and (training) else 0
        total_cross_ent += Cross_entropy_loss.item() if not type(Cross_entropy_loss) is list else sum(Cross_entropy_loss).item()
        total_cell_ent += (cell_ent_ind*2+Cell_ent.item())/3 if self.args['min_Cell_entropy'] and len(cluster_assignment)>0 else 0         
        total_Pat_ent += (Pat_ent.item()+pat_ent_ind*2)/3 if self.args['Max_Pat_Entropy'] and len(cluster_assignment)>0 else 0         
        

        # Eliminate usage of tensors in cuda.
        del data
        # When testing, save the clusters for visualization purposes
        if saveClusters:            
            indexes = Indices[index:min(index+self.args['batch_size'],len(Indices))]
            for idx, val in enumerate(indexes):
                self.dataset.save_cluster_and_attention(idx, val, save_Inductivecluster_presence, cluster_assignment,attentionVector, cluster_interaction)

    n_iter = len(list(range(0,len(Indices),self.args['batch_size'])))

    return GroundTruthLabelsAll, PredictedLabelsAll, (correct / PredictedLabelsAll[0].shape[0])/len(pred), total_loss / n_iter, total_ortho_color/n_iter, total_ortho_color/n_iter, unsup_loss, total_ortho/n_iter, ortho_ind, total_MinCUT_loss/n_iter, total_pearsonCoeffSUP, total_pearsonCoeffUNSUP, total_nearestNeighbor_loss/n_iter, total_UnsupContrast_loss/n_iter, total_UnsupContrastAcc/n_iter, total_cross_ent/n_iter, total_cell_ent/n_iter, total_Pat_ent/n_iter, Indices
