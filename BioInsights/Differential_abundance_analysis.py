import torch
import numpy as np
import itertools
from scipy import stats
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from BioInsights.add_annotation_stat import add_stat_annotation
import pandas as pd
from itertools import combinations
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
import random
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
from parallel_process import parallel_process
from BioInsights.adjust_pvalues import lsu
from BioInsights.adjust_pvalues import hochberg
from BioInsights.adjust_pvalues import holm_bonferroni
from BioInsights.adjust_pvalues import sidak

def differential_abundance_analysis(dataset, heatmapALL, labels, IndexAndClass,isTraining):        
        pvalueThreshold=1.25#pvalueThreshold=0.0000000001
        statisticalTest = [] # List of columns with statistical difference. 1.p-value, 2.Cluster step, 3.column of the heatmap.
        
        class Model(torch.nn.Module):
            def __init__(self, num_features, n_classes):
                super(Model, self).__init__()
                # self.lin1 = torch.nn.Linear(num_features, 1024, bias=True)                
                self.lin4 = torch.nn.Linear(num_features, n_classes, bias=True)

            def reset_parameters(self):
                # self.lin1.reset_parameters()                
                self.lin4.reset_parameters()

            def forward(self,features, labels):
                # UM = torch.nn.functional.relu(self.lin1(features)) # torch.cat(self.X,dim=-1)            
                return self.lin4(features)         

        class Load_Model(torch.nn.Module):
            def __init__(self,dataset):
                super(Load_Model, self).__init__()      
                model = torch.load(dataset.processed_dir_cross_validation+'model.pt')                          
                self.lin1 = model[0] 
                self.batchNorm = model[1] 
                self.lin2 = model[2]

            def reset_parameters(self):
                return
                # self.lin1.reset_parameters()                
                # self.lin4.reset_parameters()

            def forward(self,features, labels):
                UM = torch.nn.functional.relu(self.lin1(features)) # torch.cat(self.X,dim=-1)   
                UM = self.batchNorm(UM)       
                return self.lin2(UM)         
        
        class Model_NonLin(torch.nn.Module):
            def __init__(self, num_features, n_classes):
                super(Model_NonLin, self).__init__()
                self.lin1 = torch.nn.Linear(num_features, 1024, bias=True)                
                self.lin4 = torch.nn.Linear(1024, n_classes, bias=True)

            def reset_parameters(self):
                # self.lin1.reset_parameters()                
                self.lin4.reset_parameters()

            def forward(self,features, labels):
                UM = torch.nn.functional.relu(self.lin1(features)) # torch.cat(self.X,dim=-1)            
                return self.lin4(UM)      
            
            # loss = F.cross_entropy(UM, labels[:,0].long())     
            # pred = UM.max(1)[1].detach().cuda:1().numpy()

        def trainModel(model,dependentVariable,labels,model_is_loaded):
            # Initialize Model
            if not model_is_loaded:
                model.reset_parameters() 
                n_iter = 10000    
            else:
                n_iter = 5
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
            dependentVariable = torch.tensor(dependentVariable,device='cuda:1',dtype=torch.float32)
            labels = torch.tensor(labels,device='cuda:1',dtype=torch.float32)
            for i in range(n_iter):
                losses = model(dependentVariable,labels) 
                # pred = losses.max(1)[1].detach().cpu().numpy()                               
                # print('train_acc:',np.equal(pred,labels.cpu().numpy()).mean())
                # l = torch.nn.functional.cross_entropy(losses.squeeze(), labels)
                l = torch.nn.functional.cross_entropy(losses, labels.long())     
                l.backward()
                optimizer.step()
                # print(l.cuda:1().detach().numpy())
            pred = losses.max(1)[1].detach().cpu().numpy()            
            return model.eval(), np.equal(pred,labels.cpu().numpy()).mean(), np.stack([torch.nn.functional.cross_entropy(losses[i,:].unsqueeze(dim=0), labels[i].unsqueeze(dim=0).long()).detach().cpu().numpy() for i in range(labels.shape[0])])
        
        def validateModel(model, dependentVariable, labels):
            dependentVariable = torch.tensor(dependentVariable,device='cuda:1',dtype=torch.float32)
            labels = torch.tensor(labels,device='cuda:1',dtype=torch.float32)
            losses = model(dependentVariable,labels)  
            labels_c = labels.cpu().numpy()
            
            # Obtain the prediction confidence of the predicted class
            r_loss_pred_class = [i.max() for n,i in enumerate(torch.nn.functional.softmax(losses,dim=1).detach().cpu().numpy())]           
            r_loss_real_class = [i[int(labels_c[n])] for n,i in enumerate(torch.nn.functional.softmax(losses,dim=1).detach().cpu().numpy())]
            restricted_loss = [i[int(labels_c[n])] for n,i in enumerate(losses.detach().cpu().numpy())]
            return np.equal(losses.max(1)[1].detach().cpu().numpy(),labels_c).mean(), np.array(restricted_loss), np.array(r_loss_pred_class), np.array(r_loss_real_class), losses.max(1)[1].detach().cpu().numpy()
            
        def findMinimumNumberOfClusters(dependentVariable,model,labels,Unrestricted_loss,pvalueThreshold):
            # Indices to select unrestricted model.
            indices = np.array(list(range(dependentVariable.shape[1])))
            # Possible Combinations
            possible_comb = list(range(dependentVariable.shape[1]))

            # Find the minimum number of significant clusters.
            for n_significant_clusters in range(dependentVariable.shape[1]): 
                
                # Save p-value of statistical tests.
                if n_significant_clusters==0:
                    stats_indices = []
                elif n_significant_clusters==1:
                    possible_comb = sorted(range(len(stats_indices)), key=lambda i: stats_indices[i])[-7:]
                    
                # Iterate n_significant_clusters, and all their possible combinations.
                for selectedCluster in itertools.combinations(possible_comb,n_significant_clusters+1):                                
                    # Execute Restricted model
                    Restricted_dependentVariable = np.zeros(dependentVariable.shape)
                    for c in selectedCluster:
                        Restricted_dependentVariable[:,indices==c] = dependentVariable[:,indices==c]
                    ACC, Restricted_loss, pred_Confidence, real_class_confidence, pred_Label = validateModel(model,Restricted_dependentVariable,labels)
                    print('Restricted Model accuracy: ',ACC)

                    # Check if unrestricted model is significantly different to restricted model.
                    if stats.shapiro(Restricted_loss)[1]<0.05 or stats.shapiro(Unrestricted_loss)[1]<0.05:
                        # Normality is rejected
                        if len(selectedCluster)==1:
                            stats_indices.append(stats.kruskal(Restricted_loss,Unrestricted_loss)[1])
                        if stats.kruskal(Restricted_loss,Unrestricted_loss)[1]>pvalueThreshold:
                            return selectedCluster, stats.kruskal(Restricted_loss,Unrestricted_loss)[1] # This unrestricted mdel is not sinificantly different to the restricted one.
                    else: 
                        # Normality is accepted
                        if len(selectedCluster)==1:
                            stats_indices.append(stats.ttest_rel(Restricted_loss,Unrestricted_loss)[1])
                        if stats.ttest_rel(Restricted_loss,Unrestricted_loss)[1]>pvalueThreshold:
                            return selectedCluster, stats.ttest_rel(Restricted_loss,Unrestricted_loss)[1]  # This unrestricted mdel is not sinificantly different to the restricted one.

        def is_bad_combination(possible_ind, selectedCluster):
            for ind in possible_ind:
                num_pos=0
                for i in ind[1]:
                    num_pos += sum([s==i for s in selectedCluster])
                if num_pos==len(ind[1]):                            
                    return True
            return False
        
        def findSignificantClusters(dataset,dependentVariable,model,labels,Unrestricted_loss,pred_confidence_Unrestricted,pvalueThreshold):
            # Indices to select unrestricted model.
            indices = np.array(list(range(dependentVariable.shape[1])))
            # Possible Combinations
            possible_comb = list(range(dependentVariable.shape[1]))
            possible_ind = []
            Indices = list(range(dependentVariable.shape[1]))

            # Save Statistics
            stats_indices = []  
            
            # Find the minimum number of significant clusters.
            for n_significant_clusters in range(2): #dependentVariable.shape[1]                                                                 
              
                # Iterate n_significant_clusters, and all their possible combinations.
                for selectedCluster in itertools.combinations(Indices,n_significant_clusters+1):                                
                    
                    # Check if model should be checked
                    if not is_bad_combination(possible_ind, selectedCluster) and n_significant_clusters>0:
                        continue
                    
                    # Execute Restricted model
                    Restricted_dependentVariable = copy.deepcopy(dependentVariable)
                    for c in selectedCluster:
                        Restricted_dependentVariable[:,indices==c] = 0
                    ACC, Restricted_loss, pred_Confidence_restricted, real_class_confidence, pred_Label = validateModel(model,Restricted_dependentVariable,labels)
                    # print('Restricted Model accuracy: ',ACC)

                    # Check if unrestricted model is significantly different to restricted model.
                    # if stats.shapiro(Restricted_loss)[1]<0.05 or stats.shapiro(Unrestricted_loss)[1]<0.05:
                        # Normality is rejected
                        # if len(selectedCluster)==1:
                            # stats_indices.append(stats.kruskal(Restricted_loss,Unrestricted_loss)[1])                    
                    if np.mean(pred_confidence_Unrestricted)>np.mean(pred_Confidence_restricted):               
                        stats_indices.append([stats.kruskal(Restricted_loss,Unrestricted_loss)[1],selectedCluster,ACC,[i if i<=3 else 3 for i in pred_confidence_Unrestricted/pred_Confidence_restricted]])
                    else:
                        stats_indices.append([1,selectedCluster,ACC,[i if i<=3 else 3 for i in pred_confidence_Unrestricted/pred_Confidence_restricted]])
                    
                    # else: 
                        # Normality is accepted
                        # if len(selectedCluster)==1:
                            # # stats_indices.append(stats.ttest_rel(Restricted_loss,Unrestricted_loss)[1])
                        # if stats.ttest_rel(Restricted_loss,Unrestricted_loss)[1]<pvalueThreshold:
                            # return selectedCluster, stats.ttest_rel(Restricted_loss,Unrestricted_loss)[1]  # This unrestricted mdel is not sinificantly different to the restricted one.
                
                # Order Indices to show wich are the most important
                if n_significant_clusters==0:
                    Indices = sorted(range(len(stats_indices)), key=lambda i: stats_indices[i][0])[:int(len(stats_indices)*1)]    
                
                # Save p-value of statistical tests.                                
                possible_comb = sorted(range(len(stats_indices)), key=lambda i: stats_indices[i][0])[:int(len(stats_indices)*1)]
                possible_ind = [stats_indices[i] for i in possible_comb]            
                
                # Do the statistical test
                # print('BioInsights: 'stats_indices[possible_comb[0]][0])
                if stats_indices[possible_comb[0]][0]<pvalueThreshold:
                    break

            # Find the top 1 significant cluster per patient
            Top1PIRPerPatient=[]
            for i in range(Restricted_loss.shape[0]):
                Top1PIRPerPatient.append([(),0])
            for sts in stats_indices:
                if sts[0]<pvalueThreshold or (not 'Synthetic' in dataset.root):
                    for n, pat in enumerate(Top1PIRPerPatient):
                        if pat[1]<sts[3][n]:
                            pat[1]=sts[3][n]
                            pat[0]=sts[1]

            return stats_indices[possible_comb[0]][1], stats_indices[possible_comb[0]][0], stats_indices[possible_comb[0]][2],[stats_indices[i] for i in possible_comb], Top1PIRPerPatient# This unrestricted mdel is not sinificantly different to the restricted one.        

        def obtain_relevant_PIR(Top1PerPatient,topkClustersStats):
            '''
                Returns PIR statistics per patient
            '''

            # Obtain unique cell types that are significnat
            Top1PerPatientUniq = list(set([i[0] for i in Top1PerPatient]))
            top1ClustersStats=[]
            for topk in topkClustersStats:
                if any([len(set(uniq) & set(topk[1]))==len(uniq) for uniq in Top1PerPatientUniq]) or topk[0]<0.05:
                    top1ClustersStats.append(topk)

            # Create Prediction Index in Restricted Model for the statistically significant cell type combinations
            stats_sign = []
            RestrictedModelData={}
            for topk in top1ClustersStats:
                Significant_Clust_str = ''
                for S_C_S in topk[4]:
                    Significant_Clust_str += '+'+S_C_S[1]
                if 0.01<topk[0]<0.05:
                    RestrictedModelData[Significant_Clust_str[1:]+'*'] = topk[3]
                elif 0.001<topk[0]<0.01:
                    RestrictedModelData[Significant_Clust_str[1:]+'**'] = topk[3]
                elif topk[0]<0.001:
                    RestrictedModelData[Significant_Clust_str[1:]+'***'] = topk[3]
                else:
                    RestrictedModelData[Significant_Clust_str[1:]] = topk[3]

            return top1ClustersStats, RestrictedModelData

        def save_dict_to_excel(dictionary, filename):
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dictionary.items()])) 
            df.to_excel(filename)

        def showPatientPredictionConfidence(dataset,pred_Confidence, labels,Subgroups_of_patients_labels_names, Subgroups_of_patients_labels,topkClustersStats,IndexAndClass,Top1PerPatient,real_class_confidence):

            list_labels = [l[0] for l in labels[list(labels.keys())[0]]]
                        
            # Visualize prediction confidence and PIR
            pred_Confidence_dict = {}
            pred_Confidence_dict['Subject prediction confidence'] = pred_Confidence            
            save_dict_to_excel(pred_Confidence_dict, dataset.bioInsights_dir_abundance_analysis+'Prediction_Confidence.xlsx')                        
            df_PC =  pd.DataFrame.from_dict(pred_Confidence_dict)            
            df_PIR = pd.DataFrame.from_dict(RestrictedModelData)

            # Order Subgroups_of_patients_labels
            listt=list(range(max(set(Subgroups_of_patients_labels))+1))
            strt = 0
            for i in set(Subgroups_of_patients_labels):
                listt[i] = strt
                strt+=1
            Subgroups_of_patients_labels = [listt[spl] for spl in Subgroups_of_patients_labels]
            cp = sns.color_palette("bright",len(set(Subgroups_of_patients_labels)))
            Subgroups_of_patients_labels_colors = [cp[spl-1] for spl in Subgroups_of_patients_labels]
            names = list(Subgroups_of_patients_labels_names.keys()) 
            
            # Display PIR's per patient.
            plt.close()
            plt.figure()
            labels_pal = sns.cubehelix_palette(len(set(list_labels)), light=.9, dark=.1, reverse=True, start=0, rot=-2)
            labels_lut = dict(zip(map(str, list(set(list_labels))), labels_pal))        
            labels_colors = [labels_lut[str(k)] for k in list_labels]
            cm_PIR = sns.clustermap(df_PIR,col_cluster=False,row_cluster=True, xticklabels=1, vmin=1, vmax=1.3, row_colors=[Subgroups_of_patients_labels_colors,labels_colors], linewidths=0, cmap="magma")

            # plt.close()
            # plt.figure()
            # labels_pal = sns.cubehelix_palette(len(set(list_labels)), light=.9, dark=.1, reverse=True, start=0, rot=-2)
            # labels_lut = dict(zip(map(str, list(set(list_labels))), labels_pal))        
            # labels_colors = [labels_lut[str(k)] for k in list_labels]
            # cm_PIR = sns.clustermap(df_PIR,col_cluster=False,row_cluster=True, yticklabels=True, vmin=1, vmax=1.4,colors_ratio=0.01,dendrogram_ratio=0.1, row_colors=[labels_colors,Subgroups_of_patients_labels_colors], linewidths=0, cmap="magma",cbar_pos=(int(len(IndexAndClass)/20), .2, .03, .4))
            # hm = cm_PIR.ax_heatmap.get_position()            
            # plt.setp(cm_PIR.ax_heatmap.yaxis.get_majorticklabels(), fontsize=2)
            # cm_PIR.ax_heatmap.set_position([hm.x0, hm.y0, hm.width*0.1, hm.height])
            for label in range(len(set(Subgroups_of_patients_labels))):
                cm_PIR.ax_col_dendrogram.bar(0, 0, color=cp[label], label=names[label], linewidth=0)            
            cm_PIR.ax_col_dendrogram.legend(title=dataset.experiment_label[0], ncol=4,bbox_to_anchor=(0, 1), loc='upper right')
            # col = cm_PIR.ax_col_dendrogram.get_position()
            # cm_PIR.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.25, col.height*0.5])            
            # cm_PIR.ax_heatmap.set_xticklabels(cm_PIR.ax_heatmap.get_xticklabels(), rotation=80,fontsize=3)    
            plt.savefig(dataset.bioInsights_dir_abundance_analysis_global+'clustermap_Predictive_influence_ratio_per_patient_iter_{}.png'.format(str(dataset.args['epochs'])),dpi=1000)            

            # Display Prediction Index in Unrestricted Model
            plt.close()
            plt.figure()
            real_class_confidence_clrs = [(0,1,0) if rcc>1/len(set(list_labels)) else (1,0,0) for rcc in real_class_confidence]
            cm = sns.clustermap(pred_Confidence,col_cluster=False,row_cluster=True,vmin=0,vmax=1, 
                                row_colors=[labels_colors,real_class_confidence_clrs], linewidths=0, cmap="RdYlGn",
                                row_linkage=cm_PIR.dendrogram_row.linkage)#,figsize=(10,20))                                                
            # hm = cm.ax_heatmap.get_position()            
            # plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=2)
            # cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width*0.25, hm.height])
            # for label in list(set(list_labels)):
            #     cm.ax_col_dendrogram.bar(0, 0, color=labels_lut[label], label=label, linewidth=0)
            # cm.ax_col_dendrogram.legend(title=dataset.experiment_label[0], loc="best", ncol=5, bbox_to_anchor=(0.47, 0.8), bbox_transform=plt.gcf().transFigure)
            #     col = cm.ax_col_dendrogram.get_position()
            #     cm.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.25, col.height*0.5])            
            cm.savefig(dataset.bioInsights_dir_abundance_analysis+'heatmap_PredictionIndex_iter_{}.png'.format(str(dataset.args['epochs'])),dpi=1000) 

            return cm_PIR.dendrogram_row.linkage, df_PIR

        def visualizeRegressionAnalysis_(dataset, dependentVariable,clust,list_labels):
            '''
                Compare patient groups using violinplot with umann whitney with bonferrini correction
            '''
            for nn, clust_n in enumerate(clust[1]):                
                # Extract cell type abundance for each patient type
                data = {}
                for lbl in list(set(list_labels)):                    
                    data[lbl] = dependentVariable[:,clust_n][[True if l==lbl else False for l in list_labels]]

                # Check if values are equal                
                for v in data.keys():
                    if len(set(data[v])) <= 1:
                        data[v] = np.random.uniform(0,0.00000001,data[v].shape)
                
                # Display and save boxplots of abundance.
                plt.close()
                plt.figure()
                df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
                labels_pal = sns.cubehelix_palette(len(set(list_labels)), light=.9, dark=.1, reverse=True, start=0, rot=-2)                
                ax = sns.violinplot(data=df, order=list(set(list_labels)),palette=labels_pal)
                ax.tick_params(labelsize=10)
                ax.set_xticklabels(list(set(list_labels)), fontsize=12)
                ax.set_ylabel("Relative abundance of "+clust[4][nn][1], fontsize=18)                
                test_results_, test_results = add_stat_annotation(ax, data=df, x=None, y=None, order=list(set(list_labels)),
                                                box_pairs=list(combinations(list(set(list_labels)),2)),
                                                test='Mann-Whitney', text_format='star',
                                                loc='inside', verbose=0)
                ax.set_title('(p-value='+str(test_results[0].pval)+')', fontsize=18)
                plt.savefig(dataset.bioInsights_dir_abundance_analysis_global+'/Global_ViolinPlot(Pairs)_'+clust[4][nn][1]+'_iter_'+str(dataset.args['epochs']),dpi=600)                    

            # Extract cell type abundance for each patient type
            data = {'Patch characterization':[],'Relative abundance':[],dataset.experiment_label[0]:[]}       
            for lbl in list(set(list_labels)):     
                for n_c in range(len(clust[1])):  
                    for d in dependentVariable[:,clust[1][n_c]][[True if l==lbl else False for l in list_labels]]:
                        data['Patch characterization'].append(clust[4][n_c][1])            
                        data['Relative abundance'].append(d)
                        data[dataset.experiment_label[0]].append(lbl)                                
            boxpairs=[]
            for n_c in [i[1] for i in clust[4]]:  # For each microenvironment element        
                for lb in combinations(list(set(list_labels)),2): 
                    boxpairs.append(tuple([(n_c,labl) for labl in lb]))
            
            # Check if values are equal                               
            if len(set(data['Relative abundance'])) <= 1:
                data['Relative abundance'] = np.random.uniform(0,0.00000001,len(data['Relative abundance']))

            # Display and save boxplots of abundance.                
            plt.close()
            plt.figure()
            df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
            labels_pal = sns.cubehelix_palette(len(set(list_labels)), light=.9, dark=.1, reverse=True, start=0, rot=-2)
            ax = sns.violinplot(x='Patch characterization', y='Relative abundance', hue=dataset.experiment_label[0],data=df,palette=labels_pal)                
            test_results_, test_results = add_stat_annotation(ax, data=df,x='Patch characterization', y='Relative abundance',hue=dataset.experiment_label[0],
                                            box_pairs=boxpairs,
                                            test='Mann-Whitney', text_format='star',
                                            loc='inside', verbose=0)  
            ax.set_title('(p-value='+str(test_results[0].pval)+')', fontsize=18)
            plt.savefig(dataset.bioInsights_dir_abundance_analysis_global+'/Global_ViolinPlot_'+'_'.join([i[1] for i in clust[4]])+'_iter_'+str(dataset.args['epochs']),dpi=600)
            return 'Done'

        def visualizeRegressionAnalysis(dataset,dependentVariable,labels,pred_Label,topkClustersStats,IndexAndClass,top1):                                    

            list_labels = [l[0] for l in labels[list(labels.keys())[0]]]
            list_labels_int = [list(set(list_labels)).index(l) for l in list_labels]
                
            # Prepare parallel process
            dict_subjects = []
            for n, clust in enumerate(topkClustersStats):                
                dict_subjects.append({'dataset':dataset,'dependentVariable':dependentVariable,'clust':clust,'list_labels':list_labels})
            # dict_subjects= dict_subjects[:20]
            # select_patches_from_cohort
            result = parallel_process(dict_subjects,visualizeRegressionAnalysis_,use_kwargs=True,front_num=len(dict_subjects),desc='BioInsights: Save abundance differences between TMEs') 
                            
        def showGroups_Patients(RestrictedModelData,labels,dependentVariable,Top1PerPatient,top1ClustersStats,dataset,IndexAndClass):
            '''
                docstring
            '''

            # Calculate dataframe from model data
            PIR_df = pd.DataFrame.from_dict(RestrictedModelData)

            # Perform groups of patients per class.
            list_labels = [l[0] for l in labels[list(labels.keys())[0]]]
            
            # Patient Sub-grouping
            patientSubgroups = {dataset.experiment_label[0]:[],'Patient subgroup':[],'Microenvironment element':[],'PIR value':[],'Patient_index':[],'Patient_name':[],'Centroid_Values: '+'_'.join(PIR_df.keys()):[]}
            Subgroups_of_patients_labels = [0 for i in IndexAndClass]
            for lbl_n, lbl in enumerate(list(set(list_labels))):
                # Extract list of patients for a certain class
                lbl_P = PIR_df[[True if lbl==l else False for l in list_labels]]
                patient_list = [l for l in range(len(list_labels)) if lbl==list_labels[l]]
                # Perform clustering on patients from a certain class             
                labels_ = fcluster(ward(pdist(np.array(lbl_P))), t=max(int(lbl_P.shape[0]/18),1), criterion='maxclust')-1            
                labels_cluster_center = [np.array(lbl_P)[labels_==i].mean(0) for i in range(len(set(labels_)))]
                # kmeans = KMeans(n_clusters=max(int(lbl_P.shape[0]/18),1), random_state=0).fit(lbl_P)
                # Assign classes to original list of patients
                for p_n, p in enumerate(patient_list):
                    Subgroups_of_patients_labels[p] = (labels_[p_n]+(100**lbl_n))
                # Gather subgroup information in dataframe
                for subgr_lbl in range(len(set(labels_))):
                    for micro_el in lbl_P.keys():                        
                        for indx, pir_value in enumerate(lbl_P[micro_el][labels_==subgr_lbl]):                                                  
                            patientSubgroups[dataset.experiment_label[0]].append(lbl)
                            patientSubgroups['Patient subgroup'].append(subgr_lbl)
                            patientSubgroups['Microenvironment element'].append(micro_el)
                            patientSubgroups['PIR value'].append(pir_value)   
                            patientSubgroups['Patient_index'].append(lbl_P[micro_el][labels_==subgr_lbl].axes[0][indx]),
                            patientSubgroups['Patient_name'].append(IndexAndClass[lbl_P[micro_el][labels_==subgr_lbl].axes[0][indx]][0])  
                            patientSubgroups['Centroid_Values: '+'_'.join(lbl_P.keys())].append(labels_cluster_center[subgr_lbl])

            # Show heatmap with differences between kmeans centroid
            TME_Names = [k.split('*')[0] for k in PIR_df.keys()]
            TME_Names_Raw = [k for k in PIR_df.keys()]
            listOfIndicesForTME = [t[1] for t in top1ClustersStats]
            TME_Names_oneByone = []
            heatmap_TME = {}
            for TME in TME_Names:
                for T in TME.split('+'):
                    TME_Names_oneByone.append(T)
                    heatmap_TME[T] = []
            heatmap_centroid = {} 
            for PG in set(patientSubgroups[dataset.experiment_label[0]]):
                locate_PG = np.array(patientSubgroups[dataset.experiment_label[0]])==PG
                SubGroups = set(np.array(patientSubgroups['Patient subgroup'])[locate_PG])
                for SG in SubGroups:
                    locate_PG_and_PS = np.logical_and(np.array(patientSubgroups['Patient subgroup'])==SG,locate_PG)
                    Centroid_Values = np.array(patientSubgroups['Centroid_Values: '+'_'.join(PIR_df.keys())])[locate_PG_and_PS][0]
                    if len(Centroid_Values)==1:
                        heatmap_centroid[dataset.experiment_label[0]+':'+PG+'-PS'+str(SG+1)]=list(Centroid_Values)                                        
                    else:
                        heatmap_centroid[dataset.experiment_label[0]+':'+PG+'-PS'+str(SG+1)]=list(stats.zscore(Centroid_Values,nan_policy='omit'))                                        
                    indTME = 0
                    for n_TME, TME in enumerate(TME_Names):
                        for n_TME_un, TME_un in enumerate(TME.split('+')):
                            Patient_indexes = np.unique(np.array(patientSubgroups['Patient_index'])[locate_PG_and_PS])
                            heatmap_TME[TME_Names_oneByone[indTME]].append(np.mean(dependentVariable[Patient_indexes,listOfIndicesForTME[n_TME][n_TME_un]]))
                            indTME+=1

            # TOP-10 PIRs for each TME.
            topk = 10
            ALLPatient_Subgroups = list(heatmap_centroid.keys())
            topk_elements = []            
            for lbl_n, lbl in enumerate(list(set(list_labels))):
                locate_perLabel = np.array([ttt==lbl for ttt in patientSubgroups[dataset.experiment_label[0]]])
                for TME_i in PIR_df.keys():
                    PIRs_for_one_TME = np.array([ttt == TME_i for ttt in patientSubgroups['Microenvironment element']])
                    PIRS_and_label = np.logical_and(PIRs_for_one_TME,locate_perLabel)
                    list_PIRs = [[n_t,t] for n_t, t in enumerate(patientSubgroups['PIR value']) if PIRS_and_label[n_t]]
                    list_PIRs.sort(key = lambda x: x[1])
                    for sorted_V in list_PIRs[:-topk:-1]:                        
                        topk_elements.append(sorted_V[0])
            selected_Patients = {'Patient subgroup':[], 'Patient index': [], 'PIR value':[], 'TME':[], 'TME -h':[],'Patient Name':[], 'Patient class':[]} 
            for p_ind in topk_elements:
                selected_Patients['Patient subgroup'].append(patientSubgroups['Patient subgroup'][p_ind])
                selected_Patients['Patient index'].append(patientSubgroups['Patient_index'][p_ind])
                selected_Patients['PIR value'].append(patientSubgroups['PIR value'][p_ind])
                TME_ind = listOfIndicesForTME[list(PIR_df.keys()).index(patientSubgroups['Microenvironment element'][p_ind])]
                selected_Patients['TME'].append([[i for i in Top1PerPatient if i[0]==TME_ind][0]])
                selected_Patients['TME -h'].append(patientSubgroups['Microenvironment element'][p_ind].split('*')[0])
                selected_Patients['Patient Name'].append(patientSubgroups['Patient_name'][p_ind])
                selected_Patients['Patient class'].append(patientSubgroups[dataset.experiment_label[0]][p_ind])
     

            # # TOP-2 PIRs for each TME.
            # topk = 2
            # ALLPatient_Subgroups = list(heatmap_centroid.keys())
            # topk_elements = []
            # for n_TME, PIRs_for_one_TME in enumerate(np.transpose(np.array([heatmap_centroid[PS] for PS in heatmap_centroid.keys()]))):
            #     for lbl_n, lbl in enumerate(list(set(list_labels))):
            #         locate_perLabel = np.array([lbl in hc.split(':')[1] for hc in list(heatmap_centroid.keys())])
            #         sorted_indices_PIR = PIRs_for_one_TME.argsort()[::-1]
            #         for sorted_V in sorted_indices_PIR:
            #             if locate_perLabel[sorted_V]:
            #                 topk_elements.append([ALLPatient_Subgroups[sorted_V],n_TME])
            #                 break        
            # Show PIRs mean values for patient subgroups
            if len(heatmap_centroid[list(heatmap_centroid.keys())[0]])>1:
                plt.close()
                plt.figure()    
                df2 = pd.DataFrame.from_dict(heatmap_centroid).T        
                scm = sns.clustermap(df2,row_cluster=True,col_cluster=True,vmin=-2,vmax=2,xticklabels=TME_Names)
                plt.savefig(dataset.bioInsights_dir_abundance_analysis_Subgroups+'/Clustermap_SubgroupsDifferencesPIR.png',dpi=600)                                 
                # Show TME abundance differences for patient subroups highlighting top2 PIR values.
                plt.close()
                plt.figure()    
                df2 = pd.DataFrame.from_dict(heatmap_TME)        
                ax = sns.clustermap(df2,row_cluster=True,col_cluster=True,vmin=-2,vmax=2,yticklabels=list(heatmap_centroid.keys()))
                # from matplotlib.patches import Rectangle
                # ax = ax.ax_heatmap
                # for topkEL in topk_elements:            
                #     ax.add_patch(Rectangle((topkEL[1], ALLPatient_Subgroups.index(topkEL[0])), 1, 1, fill=False, edgecolor='blue',  lw=3))                
                plt.savefig(dataset.bioInsights_dir_abundance_analysis_Subgroups+'/Clustermap_SubgroupsDifferencesTME_iter_{}.png'.format(str(dataset.args['epochs'])),dpi=600)
            # Show PIR's differences between patient subgroups
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in patientSubgroups.items()])) 
            df.to_excel(dataset.bioInsights_dir_abundance_analysis_Subgroups+'Subgroup_Information_{}.xlsx'.format(str(dataset.args['epochs'])))
            # # Return pairs of patients that show mechanisms.
            # selected_Patients = {'Patient subgroup':[], 'Patient index': [], 'PIR value':[], 'TME':[], 'TME -h':[],'Patient Name':[], 'Patient class':[]} 
            # for topkEL in topk_elements[:np.array([t[0]<0.05 for t in top1ClustersStats]).sum()*2]:
            #     PS_Located = np.array(df['Patient subgroup']==int(topkEL[0].split('-PS')[1])-1)
            #     clss_Located = np.array(df[dataset.experiment_label[0]]==topkEL[0].split(':')[-1].split('-PS')[0]) 
            #     TME_Located = np.array(df['Microenvironment element']==TME_Names_Raw[topkEL[1]])
            #     Patient_PIR_max_Val = df['PIR value'][np.logical_and(np.logical_and(PS_Located,clss_Located),TME_Located)].max()
            #     PIR_max_val = np.array(df['PIR value']==Patient_PIR_max_Val)
            #     patient_indx = df['Patient_index'][np.logical_and(np.logical_and(PS_Located,clss_Located),TME_Located)].array
            #     for p_ind in patient_indx:
            #         selected_Patients['Patient subgroup'].append(int(topkEL[0].split('-PS')[1]))
            #         selected_Patients['Patient index'].append(p_ind)
            #         selected_Patients['PIR value'].append(Patient_PIR_max_Val)
            #         selected_Patients['TME'].append([[t for t in Top1PerPatient if t[0]==listOfIndicesForTME[topkEL[1]]][0][2]])
            #         selected_Patients['TME -h'].append(TME_Names_Raw[topkEL[1]])
            #         selected_Patients['Patient Name'].append(IndexAndClass[p_ind][0])
            #         selected_Patients['Patient class'].append(topkEL[0].split(':')[-1].split('-PS')[0])

            # # Intra class differences            
            # for clss in list(set(list_labels)):
            #     for micro_elem in set(patientSubgroups['Microenvironment element']):                    
            #         for subgrp_0 in set(np.array(patientSubgroups['Patient subgroup'])[np.array(patientSubgroups[dataset.experiment_label[0]])==clss]):
            #             for subgrp_1 in set(np.array(patientSubgroups['Patient subgroup'])[np.array(patientSubgroups[dataset.experiment_label[0]])==clss]):                        
            #                 clss_d = np.array(df[dataset.experiment_label[0]]==clss)
            #                 micro_elem_d = np.array(df['Microenvironment element']==micro_elem)
            #                 subgrp_0_d = np.array(df['Patient subgroup']==subgrp_0)
            #                 subgrp_1_d = np.array(df['Patient subgroup']==subgrp_1)
            #                 s1 = np.logical_and(np.logical_and(clss_d,micro_elem_d),subgrp_0_d)
            #                 s2 = np.logical_and(np.logical_and(clss_d,micro_elem_d),subgrp_1_d)                                                        
            #                 try:
            #                     p_val = mannwhitneyu(df['PIR value'][s1],df['PIR value'][s2])
            #                 except:
            #                     p_val=[1,1]
            #                 if p_val[1]<0.001:
            #                     bxplt = {'Patient subgroup':['PS'+str(subgrp_0)]*len(list(df['PIR value'][s1]))+['PS'+str(subgrp_1)]*len(list(df['PIR value'][s2])),'{} PIR values per patient'.format(micro_elem):list(df['PIR value'][s1])+list(df['PIR value'][s2])}
            #                     df_b = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in bxplt.items() ])) 
            #                     # Display and save boxplots of abundance.                
            #                     plt.close()
            #                     plt.figure()
            #                     ax = sns.boxplot(x='Patient subgroup', y='{} PIR values per patient'.format(micro_elem),data=df_b,order=['PS'+str(subgrp_0),'PS'+str(subgrp_1)])#, hue=dataset.experiment_label[0],data=df_b)                
            #                     test_results = add_stat_annotation(ax, data=df_b,x='Patient subgroup', y='{} PIR values per patient'.format(micro_elem),order=['PS'+str(subgrp_0),'PS'+str(subgrp_1)],
            #                                                     box_pairs=[('PS'+str(subgrp_0),'PS'+str(subgrp_1))],
            #                                                     test='Mann-Whitney', text_format='star',
            #                                                     loc='inside', verbose=2)                                                                                                                
            #                     plt.title('Intraclass Differences {} {} pval{}'.format(clss,micro_elem,np.round(p_val[1],4)))
            #                     plt.savefig(dataset.bioInsights_dir_abundance_analysis_Subgroups+'IntraClassDiff_{}_{}.png'.format(clss,micro_elem),dpi=600)
                                

            # # Inter class differences            
            # for micro_elem in set(patientSubgroups['Microenvironment element']):                    
            #     for subgrp_0 in set(np.array(patientSubgroups['Patient subgroup'])[np.array(patientSubgroups[dataset.experiment_label[0]])==clss]):
            #         for subgrp_1 in set(np.array(patientSubgroups['Patient subgroup'])[np.array(patientSubgroups[dataset.experiment_label[0]])==clss]):                        
            #             clss_d = np.array(df[dataset.experiment_label[0]]==clss)
            #             micro_elem_d = np.array(df['Microenvironment element']==micro_elem)
            #             subgrp_0_d = np.array(df['Patient subgroup']==subgrp_0)
            #             subgrp_1_d = np.array(df['Patient subgroup']==subgrp_1)
            #             s1 = np.logical_and(np.logical_and(clss_d,micro_elem_d),subgrp_0_d)
            #             s2 = np.logical_and(np.logical_and(clss_d,micro_elem_d),subgrp_1_d)                                                        
            #             p_val = mannwhitneyu(df['PIR value'][s1],df['PIR value'][s2])
            #             if p_val[1]<0.001:
            #                 bxplt = {'PS'+str(subgrp_0):df['PIR value'][s1],
            #                             'PS'+str(subgrp_1):df['PIR value'][s2]}
            #                 df_b = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in bxplt.items() ])) 
            #                 plt.close()
            #                 fig = plt.figure()
            #                 ax = fig.add_subplot(111)
            #                 ax.boxplot([np.array(df['PIR value'][s1]),np.array(df['PIR value'][s2])])
            #                 ax.set_xticklabels(['PS'+str(subgrp_0), 'PS'+str(subgrp_1)])                                
            #                 plt.title('IntraClassDiff_{}_{}_pval{}.png'.format(clss,micro_elem,np.round(p_val[1],4)))
            #                 plt.savefig(dataset.bioInsights_dir_abundance_analysis+'IntraClassDiff_{}_{}.png'.format(clss,micro_elem),dpi=600)

      
            # Show PIR's differences between patient subgroups
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in selected_Patients.items()])) 
            df.to_excel(dataset.bioInsights_dir_abundance_analysis_Subgroups+'Patients_toSHow_{}.xlsx'.format(str(dataset.args['epochs'])))

            # # Return best patient to show for each subgroup of patients.
            # Patient_subgroup = []
            # for clss in list(set(list_labels)):
            #     for subgrp_0 in set(np.array(patientSubgroups['Patient subgroup'])[np.array(patientSubgroups[dataset.experiment_label[0]])==clss]):                    
            #         subgrp_0_d = np.array(df['Patient subgroup']==subgrp_0)
            #         clss_d = np.array(df[dataset.experiment_label[0]]==clss)                    
            #         Patient_PIR_max_Val = df['PIR value'][np.logical_and(clss_d,subgrp_0_d)].max()
            #         Patient_subgroup.append(list(df['Patient_index'][df['PIR value']==Patient_PIR_max_Val])[0])

            print('BioInsights: Extraction of most important PIRs per TME')
            return selected_Patients, heatmap_centroid, Subgroups_of_patients_labels, Top1PerPatient

        def Initialize_model(dataset, heatmapALL,labels):
            '''
            '''
            # Initialize Variables            
            dependentVariable = np.concatenate(tuple([heatmapALL[h] for h in heatmapALL]),1)
            dependentVariable = (dependentVariable-dependentVariable.mean(0))/(dependentVariable.std(0)+1e-16)
            labels = np.array(labels[list(labels)[0]])

            # Initialize Model, and obtain unrestricted model.
            # model = Load_Model(dependentVariable.shape[1],labels.max()+1) 
            labels_names = [l[0] for l in labels] 
            labels_int = [list(set(labels_names)).index(l) for l in labels_names]       
            loadmodel_pls= False
            if loadmodel_pls:
                model = Load_Model(dataset)
                model.to('cuda:1')
            else:
                model = Model(dependentVariable.shape[1],len(set(labels_names)))             
                model.to('cuda:1')
            model.train()
            model, ACC, Unrestricted_loss = trainModel(model,dependentVariable,labels_int,loadmodel_pls)     
            ACC, Unrestricted_loss, pred_Confidence, real_class_confidence, pred_Label = validateModel(model, dependentVariable, labels_int)
            print('BioInsights: Unrestricted Model accuracy is ',ACC)
        
            return dependentVariable,model,labels_int,Unrestricted_loss,pred_Confidence, pred_Label, real_class_confidence
        
        def adjust_pvalue(Names,p_values,filename):
            Pvalue_adjust = {'Names':Names}
            Pvalue_adjust['lsu_values'] = lsu(np.array(p_values),alpha=0.05)[1]
            Pvalue_adjust['hochberg'] = hochberg(np.array(p_values))
            Pvalue_adjust['holm_bonferroni'] = holm_bonferroni(np.array(p_values))
            Pvalue_adjust['sidak'] = sidak(np.array(p_values))
            Pvalue_adjust['Original_pvalues'] = np.array(p_values)
            df = pd.DataFrame(Pvalue_adjust)
            df.to_excel(filename)  

        def TranslateIndexToCellTypes(dataset, topkClustersStats, heatmapALL, significantClusters):
        
            for topk in topkClustersStats:
                topk.append([])

            # Obtain the real number of the cluster
            statisticalTest = []
            index = 0
            for n,clusterLevel in enumerate(heatmapALL):
                for c in range(int(clusterLevel)):
                    if any([s==index for s in significantClusters]):
                        statisticalTest.append([pvalue_restrictedModel,clusterLevel,c])                    
                    for topk in topkClustersStats:
                        if topk[0]<0.05 and n==0 and c==0:
                            Top1PerPatient.append([topk[1],1])
                        if any([f==index for f in topk[1]]):
                            if n==0:
                                topk[4].append([pvalue_restrictedModel,'P'+str(c+1),c])
                            elif n==1:
                                topk[4].append([pvalue_restrictedModel,'N'+str(c+1),c])
                            elif n==2:
                                topk[4].append([pvalue_restrictedModel,'A'+str(c+1),c])    
                    for top1 in Top1PerPatient: 
                        if any([s==index for s in top1[0]]):           
                            top1.append([clusterLevel,c])
                    index+=1
            df = pd.DataFrame(topkClustersStats)
            df.to_excel(dataset.bioInsights_dir_abundance_analysis+"Significance_of_TMEs.xlsx")  
            print('BioInsights: Relevant TME/s '+' '.join([i[1] for i in topkClustersStats[0][4]])+' (p-val = '+str(round(topkClustersStats[0][4][0][0],4))+')')
            adjust_pvalue(Names=[topk[4] for topk in topkClustersStats],p_values=[topk[0] for topk in topkClustersStats],filename=dataset.bioInsights_dir_abundance_analysis+"Significance_of_TMEs_adjusted.xlsx")
            return Top1PerPatient, topkClustersStats, statisticalTest

        # Initialize regression model. 
        dependentVariable,model,labels_int,Unrestricted_loss,pred_Confidence, pred_Label,real_class_confidence = Initialize_model(dataset,heatmapALL,labels)

        # Find significant clusters using the model
        significantClusters, pvalue_restrictedModel, ACCRestricted, topkClustersStats, Top1PerPatient = findSignificantClusters(dataset,dependentVariable,model,labels_int,Unrestricted_loss,pred_Confidence,pvalueThreshold)        
        
        # Translate index to cell types.
        Top1PerPatient, topkClustersStats, significantClustersStats  = TranslateIndexToCellTypes(dataset,topkClustersStats, heatmapALL, significantClusters)                

        if not isTraining:
            # Visualize Regression analysis
            visualizeRegressionAnalysis(dataset,dependentVariable,labels,pred_Label,topkClustersStats,IndexAndClass,Top1PerPatient)

            # Obtain relevant PIRs
            topKClustersStats, RestrictedModelData = obtain_relevant_PIR(Top1PerPatient,topkClustersStats)

            # Calculate distinct groups of patients. To show intra-class heterogeneity            
            patient_Ineach_subgroup, Subgroups_of_patients_labels_names, Subgroups_of_patients_labels, Top1PerPatient = showGroups_Patients(RestrictedModelData,labels,dependentVariable,Top1PerPatient,topKClustersStats,dataset,IndexAndClass)

            # Show patient Prediction Confidence
            linkage, PIR_df = showPatientPredictionConfidence(dataset,pred_Confidence,labels,Subgroups_of_patients_labels_names, Subgroups_of_patients_labels,topKClustersStats,IndexAndClass,Top1PerPatient,real_class_confidence)            
        else:
            patient_Ineach_subgroup = []
        
        return significantClustersStats, Unrestricted_loss, Top1PerPatient, patient_Ineach_subgroup, real_class_confidence