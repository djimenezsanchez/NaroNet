import torch
import torch.nn.functional as F
import NaroNet.NaroNet_model.pooling as pooling
import NaroNet.NaroNet_model.loss as loss
import NaroNet.NaroNet_model.GNN as GNN
import NaroNet.utils.utilz
import numpy as np

class NaroNet_model(torch.nn.Module):
    def __init__(self, num_features, labels, hidden, num_nodes, clusts, args):
        super(NaroNet_model, self).__init__()
        # Initialization
        if  args['LSTM']:
            self.features = hidden
        else:
            self.features = num_features        
        self.args = args         
        self.hidden = hidden                               
        
        # Phenotypes     
        if args['Phenotypes']:
            if args['LSTM']:
                self.LSTM_phenoClust = GNN.LSTM(num_features, hidden, args)                
            if args['GLORE']:                
                self.GloRe_phenoClust = GNN.GloRe_Unit(self.features, hidden, normalize=True)
                
        # FCN  
        self.pheno_blocks_clust = GNN.phenoNN8(self.features, hidden,clusts[0], args, mode='Multiplication')
                        
        # First Graph Neural Network Embedding 
        if args['LSTM']:
            self.LSTM_GNN1_Clust = GNN.LSTM(num_features, hidden, args)
            self.LSTM_GNN1_Emb = GNN.LSTM(num_features, hidden, args)
        if args['GLORE']:
            self.GloRe_GNN1_Clust = GNN.GloRe_Unit(self.features, hidden, normalize=True)
            self.GloRe_GNN1_Emb = GNN.GloRe_Unit(self.features, hidden, normalize=True)
        if args['modeltype']=='SAGE':
            self.embed_pool_block_emb = GNN.GNN(self.features, hidden, hidden, args, args['n-hops'], mode='SparseMultiplication')
            self.embed_pool_block_clust = GNN.GNN(self.features, hidden,  clusts[1], args,args['n-hops'], mode='SparseMultiplication')               
            
        # Second Graph Neural Network Embedding
        if args['modeltype']=='SAGE':
            self.embed_pool_block2_clust = GNN.GNN(hidden, hidden,  clusts[2], args,1, mode='Multiplication')
        
        # # Feature SAGPOOL
        # self.X1_SAGPOOL = GNN.phenoNN(hidden, hidden, 1, args, mode='Multiplication')
        # self.X2_SAGPOOL = GNN.phenoNN(hidden, hidden, 1, args, mode='Multiplication')

        # Unsupervised learning
        if args['UnsupContrast']:
            self.lin1_unsupA = torch.nn.Linear(clusts[0],hidden+hidden)
            self.lin1_unsupB = torch.nn.Linear(hidden+hidden,hidden)
            self.lin2_unsupA = torch.nn.Linear(clusts[1],hidden+hidden)
            self.lin2_unsupB = torch.nn.Linear(hidden+hidden,hidden)
            self.lin3_unsupA = torch.nn.Linear(clusts[2],hidden+hidden)
            self.lin3_unsupB = torch.nn.Linear(hidden+hidden,hidden)
            self.lr_unsup = torch.nn.Parameter(torch.ones(1))

        # Fully connected Layer
        self.lin1_1 = torch.nn.Linear(clusts[0] + clusts[1] + clusts[2], hidden+hidden+hidden+hidden, bias=True)          
        self.BNLast_1 = torch.nn.BatchNorm1d(hidden+hidden+hidden+hidden,track_running_stats=False)
        self.lin2_1 = torch.nn.Linear(hidden+hidden+hidden+hidden, int(max(labels[0])) if len(labels[0])>5 else len(labels[0]), bias=True)
        if len(args['experiment_Label'])>1:
            self.lin1_2 = torch.nn.Linear(clusts[0] + clusts[1] + clusts[2], hidden+hidden+hidden+hidden, bias=True)             
            self.BNLast_2 = torch.nn.BatchNorm1d(hidden+hidden+hidden+hidden,track_running_stats=False)
            self.lin2_2 = torch.nn.Linear(hidden+hidden+hidden+hidden, int(max(labels[1])) if len(labels[1])>5 else len(labels[1]), bias=True)
        if len(args['experiment_Label'])>2:
            self.lin1_3 = torch.nn.Linear(clusts[0] + clusts[1] + clusts[2], hidden+hidden+hidden+hidden, bias=True)             
            self.BNLast_3 = torch.nn.BatchNorm1d(hidden+hidden+hidden+hidden,track_running_stats=False)
            self.lin2_3 = torch.nn.Linear(hidden+hidden+hidden+hidden, int(max(labels[2])) if len(labels[2])>5 else len(labels[2]), bias=True)
        if len(args['experiment_Label'])>3:
            self.lin1_4 = torch.nn.Linear(clusts[0] + clusts[1] + clusts[2], hidden+hidden+hidden+hidden, bias=True)                                 
            self.BNLast_4 = torch.nn.BatchNorm1d(hidden+hidden+hidden+hidden,track_running_stats=False)        
            self.lin2_4 = torch.nn.Linear(hidden+hidden+hidden+hidden, int(max(labels[3])) if len(labels[3])>5 else len(labels[3]), bias=True)

    def reset_parameters(self):        
                
        # Phenotypes
        if self.args['Phenotypes']:        
            if self.args['LSTM']:
                for name, module in self.LSTM_phenoClust.named_children():
                    module.reset_parameters()                
            if self.args['GLORE']:
                self.GloRe_phenoClust.reset_parameters()                                            
            if self.args['DeepSimple']:
                for pheno in self.pheno_blocks_clust:
                    pheno.reset_parameters()
            else:
                self.pheno_blocks_clust.reset_parameters()    
                
        # First GNN        
        if self.args['LSTM']:
            for name, module in self.LSTM_GNN1_Clust.named_children():
                module.reset_parameters()
            for name, module in self.LSTM_GNN1_Emb.named_children():
                module.reset_parameters()  
        if self.args['GLORE']:
            self.GloRe_GNN1_Emb.reset_parameters()                
            self.GloRe_GNN1_Clust.reset_parameters()
        self.embed_pool_block_emb.reset_parameters()      
        self.embed_pool_block_clust.reset_parameters()                  
        
        # Second GNN        
        self.embed_pool_block2_clust.reset_parameters()              

        # # SAGPOOL
        # self.X1_SAGPOOL.reset_parameters()
        # self.X2_SAGPOOL.reset_parameters()
        
        # SUP CON LOSS
        # self.SUPCONlin.reset_parameters()

        # Unsupervised learning
        if self.args['UnsupContrast']:
            self.lin1_unsupA.reset_parameters()
            self.lin1_unsupB.reset_parameters()
            self.lin2_unsupA.reset_parameters()
            self.lin2_unsupB.reset_parameters()
            self.lin3_unsupA.reset_parameters()
            self.lin3_unsupB.reset_parameters()
            self.lr_unsup = torch.nn.Parameter(torch.ones(1))

        if self.args['ortho']:
            self.lr_ortho = torch.nn.Parameter(torch.ones(1))

        if self.args['orthoColor']:
            self.lr_orthoColor = torch.nn.Parameter(torch.ones(1))

        if self.args['F-test']:
            self.ftest_lin.reset_parameters()
            self.ftest_BNLast = torch.nn.BatchNorm1d(self.hidden+self.hidden,track_running_stats=False)
            self.ftest_lin2.reset_parameters()


        # Fully connected Layer               
        self.lin1_1.reset_parameters()
        self.BNLast_1 = torch.nn.BatchNorm1d(self.hidden+self.hidden+self.hidden+self.hidden,track_running_stats=False)
        self.lin2_1.reset_parameters()
        if len(self.args['experiment_Label'])>1:
            self.lin1_2.reset_parameters()           
            self.BNLast_2 = torch.nn.BatchNorm1d(self.hidden+self.hidden+self.hidden+self.hidden,track_running_stats=False)
            self.lin2_2.reset_parameters()
        if len(self.args['experiment_Label'])>2:
            self.lin1_3.reset_parameters()
            self.BNLast_3 = torch.nn.BatchNorm1d(self.hidden+self.hidden+self.hidden+self.hidden,track_running_stats=False)
            self.lin2_3.reset_parameters()
        if len(self.args['experiment_Label'])>3:
            self.lin1_4.reset_parameters()
            self.BNLast_4 = torch.nn.BatchNorm1d(self.hidden+self.hidden+self.hidden+self.hidden,track_running_stats=False)
            self.lin2_4.reset_parameters()
                    
    def softmaxToClst(self,s,args,device):
        s=F.softmax(s,dim=-1)
        alpha=0.1
        if args['1cell1cluster']:
            # s = torch.where(s>=s.max(-1).values.unsqueeze(-1).repeat(1,1,s.shape[2]), alpha*s+(1-alpha), torch.tensor([0],dtype=torch.float32).to(device))
            s = torch.where(s>=s.max(-1).values.unsqueeze(-1).repeat(1,1,s.shape[2]), s, s)
        s = torch.where(s>=args['attntnThreshold'], s, s*0)
        # s[s<args['attntnThreshold']]=s[s<args['attntnThreshold']]*0.1
        # Gamma Correction: 0.5
        # s = s**1.5
        # s = torch.where(s>3*args['attntnThreshold']/s.shape[2], s, torch.tensor([0],dtype=torch.float32).to(device))
        # print('Cluster Confidence:', s.max(-1).values.mean())
        # print('Activated Nodes:', (s.max(-1)[0]>0).sum().detach().cpu().numpy()/(s.shape[0]*s.shape[1]))
        return s#/s.sum(dim=-1,keepdim=True)

    def sigmoidToAttn(self,s,args,device):
        s=torch.sigmoid(s)
        # s=F.softmax(s,dim=-1)
        if args['1cell1cluster']:
            s = torch.where(s>=s.max(-1).values.unsqueeze(-1).repeat(1,1,s.shape[2]), s, torch.tensor([0],dtype=torch.float32).to(device))        
        s = torch.where(s>=args['attntnThreshold'], s, s*0)
        # s[-1][s[-1]<args['attntnThreshold']]=s[-1][s[-1]<args['attntnThreshold']]*0.1
        # s = torch.where(s>=s.max(-1).values.unsqueeze(-1).repeat(1,1,s.shape[2]), s, torch.tensor([0],dtype=torch.float32).to(device))
        # s = torch.where(s>args['attntnThreshold'], s, torch.tensor([0],dtype=torch.float32).to(device))
        return s#/s.sum(dim=-1,keepdim=True)
    
    def poolingToClst(self,s,device,num_nodes):
        Spatient = torch.zeros(s.shape[0],s.shape[-1],dtype=torch.float32).to(device)
        for i in range(s.shape[0]):
            Spatient[i,:] = (s[i,:num_nodes[i],:].sum(0)/num_nodes[i])   
        return Spatient

    def SAGPOOL(self,score,x,num_nodes):
        score=F.softmax(score,dim=1)        
        for b in range(x.shape[0]):
            s,idx = score[b,:,0].sort(-1,descending=True)
            x[b:,:int(num_nodes),:] = score[b,idx[:int(num_nodes)],:]*x[b,idx[:int(num_nodes)],:]
        return x[:,:int(num_nodes),:]

    def ObtainPhenotypesClustering(self, x, data, device, args):                        
        if args['DeepSimple']: # Multiple Neural Networks but they have few parameters...
            self.s=[]
            for pheno in self.pheno_blocks_clust:
                self.s.append(pheno(x, data.edge_index, device, data.num_nodes,args))
            self.s = torch.stack(self.s,dim=-1)
            self.s = [torch.squeeze(self.s,dim=-2)]
        else:
            self.s=[self.pheno_blocks_clust(x, data.edge_index, device, data.num_nodes,args)]        
        return self.s

    def ClassifyPatients(self, args):
        x = []
        
        # Fully connected Layer               
        x.append(F.relu(self.lin1_1(torch.cat(self.S,dim=-1))))
        if args['Batch_Normalization']:
            x[0] = self.BNLast_1(x[0]) if x[0].shape[0]>1 else x[0]                        
        x[0] = F.dropout(x[0], p=args['dropoutRate'], training=self.training)
        x[0] = self.lin2_1(x[0])          
        if len(self.args['experiment_Label'])>1:
            x.append(F.relu(self.lin1_2(torch.cat(self.S,dim=-1))))
            if args['Batch_Normalization']:
                x[1] = self.BNLast_2(x[1]) if x[1].shape[0]>1 else x[1]                        
            x[1] = F.dropout(x[1], p=args['dropoutRate'], training=self.training)
            x[1] = self.lin2_2(x[1])  
        if len(self.args['experiment_Label'])>2:
            x.append(F.relu(self.lin1_3(torch.cat(self.S,dim=-1))))
            if args['Batch_Normalization']:
                x[2] = self.BNLast_3(x[2]) if x[2].shape[0]>1 else x[2]                        
            x[2] = F.dropout(x[2], p=args['dropoutRate'], training=self.training)
            x[2] = self.lin2_3(x[2])  
        if len(self.args['experiment_Label'])>3:
            x.append(F.relu(self.lin1_4(torch.cat(self.S,dim=-1))))
            if args['Batch_Normalization']:
                x[3] = self.BNLast_4(x[3]) if x[3].shape[0]>1 else x[3]                        
            x[3] = F.dropout(x[3], p=args['dropoutRate'], training=self.training)
            x[3] = self.lin2_4(x[3])  
                
        return x

    def TissueCommunitiesInter_forward(self, data, doClustering, device, args):
        
        # GNN Forward to obtain clusters.
        self.s.append(self.embed_pool_block2_clust(data.x, data.edge_index, device, data.num_nodes,args))                            
        
        # Obtain Cell Entropy Loss
        ortho_color, pearsonCoeffSUP, pearsonCoeffUNSUP, ortho, cell_ent_loss = loss.ortho_and_mincut_loss(data,F.softmax(self.s[-1],dim=2), args, np.ones(self.s[-1].shape[0],dtype=int)*self.s[-1].shape[1],device)  if args['ClusteringOrAttention'] else loss.ortho_and_mincut_loss(data,torch.sigmoid(self.s[-1]),args, np.ones(self.s[-1].shape[0],dtype=int)*self.s[-1].shape[1],device)                    

        # Apply Softmax to cluster assignment             
        self.s[-1] = self.softmaxToClst(self.s[-1],args,device) if args['ClusteringOrAttention'] else self.sigmoidToAttn(self.s[-1],args,device)        
        
        # Node-Pooling 
        data.x, data.edge_index, minCut = pooling.Sparse_Pooling(data.y,data.x, data.edge_index, self.s[-1], device, args,self)  # Regions                      
        
        # Obtain Patient Concentration for each cluster
        self.S.append(self.poolingToClst(self.s[-1],device,[self.s[-1].shape[1]]*self.s[-1].shape[0]))                        
        
        # Obtain Patient Entropy Loss
        pat_ent = loss.pat_loss(self.S[-1],args, data.num_nodes,device)      

        self.s_interaction.append(data.edge_index) # Extract interactions between 2nd order Tissue-communities
        self.XS.append(data.x.max(-1)[0]) # Extract activations from 2nd order Tissue-communities
        self.s[-1] = self.s[-1].to('cpu') # Eliminate cluser assignment from GPU
        return data.x, data.edge_index, ortho_color, pearsonCoeffSUP, pearsonCoeffUNSUP, minCut, ortho, cell_ent_loss, pat_ent

    def TissueCommunities_forward(self, data, doClustering, device, args):
        if args['LSTM']:
            self.s.append(self.embed_pool_block_clust(self.LSTM_GNN1_Clust(data.x,device), data.edge_index, device, data.num_nodes,args))
            data.x = self.embed_pool_block_emb(self.LSTM_GNN1_Emb(data.x,device), data.edge_index, device, data.num_nodes,args)                      
        else:
            if args['GLORE']:    
                self.s.append(self.embed_pool_block_clust(self.GloRe_GNN1_Clust(data.x), data.edge_index, device, data.num_nodes,args))                    
                data.x = self.embed_pool_block_emb(self.GloRe_GNN1_Clust(data.x), data.edge_index, device, data.num_nodes,args)                                          
            else:
                self.s.append(self.embed_pool_block_clust(data.x, data.edge_index, device, data.num_nodes,args))                    
                data.x = self.embed_pool_block_emb(data.x, data.edge_index, device, data.num_nodes,args)                      
        
        # Obtain Cell Entropy Loss
        ortho_color1, pearsonCoeffSUP1, pearsonCoeffUnsup1, ortho1, cell_ent_loss1 = loss.ortho_and_mincut_loss(data,F.softmax(self.s[-1],dim=2),args, data.num_nodes,device) if args['ClusteringOrAttention'] else loss.ortho_and_mincut_loss(data,torch.sigmoid(self.s[-1]),args, data.num_nodes,device)
                
        # Apply Softmax to cluster assignment       
        self.s[-1] = self.softmaxToClst(self.s[-1],args,device) if args['ClusteringOrAttention'] else self.sigmoidToAttn(self.s[-1],args,device)   

        # Node-Pooling     
        data.x, data.edge_index, minCut1 = pooling.Dense_Pooling(data.y,data.x, data.edge_index, self.s[-1], device, data.num_nodes,doClustering, args,self, True)  # Tissue-Communities                                             
        
        # Obtain Patient Concentration for each cluster
        self.S.append(self.poolingToClst(self.s[-1],device,data.num_nodes))            
        
        # Obtain Patient Entropy Loss
        pat_ent1 = loss.pat_loss(self.S[-1],args, data.num_nodes,device)      

        self.s_interaction.append(data.edge_index)
        self.XS.append(data.x.max(-1)[0])
        
        # self.NN_loss += -self.s[-1].max(-1).values.mean()
        self.s[-1] = self.s[-1].to('cpu')
        return data.x, data.edge_index, ortho_color1, pearsonCoeffSUP1, pearsonCoeffUnsup1, minCut1, ortho1, cell_ent_loss1,pat_ent1

    def Phenotypes_forward(self, data, doClustering, device, args):
        if args['LSTM']:
            self.s, self.S = self.ObtainPhenotypesClustering(data.x,data, device, args)                        
        else:
            if args['GLORE']:    
                self.s = self.ObtainPhenotypesClustering(self.GloRe_phenoClust(data.x),data, device, args)                                           
            else:
                self.s = self.ObtainPhenotypesClustering(data.x,data, device, args)                           
               

        # Obtain Cell Entropy Loss
        ortho_color0, pearsonCoeffSUP0, pearsonCoeffUnsup0, ortho0, cell_ent_loss0 = loss.ortho_and_mincut_loss(data,F.softmax(self.s[0],dim=2),args, data.num_nodes,device) if args['ClusteringOrAttention'] else loss.ortho_and_mincut_loss(data,torch.sigmoid(self.s[0]),args, data.num_nodes,device)
                        
        # Apply Softmax to cluster assignment
        self.s[-1] = self.softmaxToClst(self.s[-1],args,device) if args['ClusteringOrAttention'] else self.sigmoidToAttn(self.s[-1],args,device)                        
        
        # Calculate Clusters Ifñormatioñ
        self.XS, self.s_interaction, minCut0 = pooling.Dense_Pooling(data.y,data.x, data.edge_index, self.s[0], device, data.num_nodes,doClustering, args, self, False)  # Phenotyep                                                                                    
        
        # Obtain Patient Concentration for each cluster
        self.S = [self.poolingToClst(self.s[-1],device,data.num_nodes)]
        
        # Obtain Patient Entropy Loss
        pat_ent0 = loss.pat_loss(self.S[-1],args, data.num_nodes,device)      

        self.s_interaction = [self.s_interaction.to('cpu')]
        # self.NN_loss = -self.s[0].max(-1).values.mean()
        self.s[0] = self.s[0].to('cpu')
    # self.X = self.X1_SAGPOOL(self.X, data.edge_index, device, data.num_nodes,args)
    # a = self.X1_SAGPOOL(data.x, data.edge_index, device, data.num_nodes,args)
    # self.X = [self.SAGPOOL(a,data.x,args['clusters1']).max(-1)[0]]
    # self.X = [self.X.max(-2)[0]]
    # self.X[0] =self.X1_SAGPOOL(self.X[0], data.edge_index, device, data.num_nodes,args)[:,:int(args['clusters1']),0]
        self.XS= [self.XS.max(-1)[0]]
        return ortho_color0, pearsonCoeffSUP0, pearsonCoeffUnsup0, minCut0, ortho0, cell_ent_loss0,pat_ent0

    def forward(self, data, device, saveClusters, trainClustering, doClustering,index,Indices,labels,args):
        
        if doClustering:                                
            # Obtain Phenotypes
            if args['Phenotypes']:
                ortho_color0, pearsonCoeffSUP0, pearsonCoeffUnsup0, minCut0, ortho0, cell_ent_loss0, pat_ent0 = self.Phenotypes_forward(data, doClustering, device, args)
            else:
                self.s = [] 
                self.S = []
                self.s_interaction = []
                self.XS = []
            
            # Obtain Tissue-communities                        
            data.x, data.edge_index, ortho_color1, pearsonCoeffSUP1, pearsonCoeffUnsup1, minCut1, ortho1, cell_ent_loss1, pat_ent1 = self.TissueCommunities_forward(data, doClustering, device, args)
            
            # Return Phenotypes and Tissue-Communities
            if args['Phenotypes']:
                return data, ortho_color0, ortho_color1, minCut0, minCut1, ortho0, ortho1, cell_ent_loss0, cell_ent_loss1, pat_ent0, pat_ent1, pearsonCoeffSUP0+pearsonCoeffSUP1, pearsonCoeffUnsup0+pearsonCoeffUnsup1, self.s            
            else:
                return data,ortho_color1, minCut1, ortho1, pearsonCoeffSUP1, pearsonCoeffUnsup1, self.s            
        
        else:        
            # Obtain Tissue-Communities Interactions
           data.x, data.edge_index, ortho_color2, pearsonCoeffSUP2, pearsonCoeffUNSUP2, minCut2, ortho2,  cell_ent_loss2, pat_ent2 = self.TissueCommunitiesInter_forward(data, doClustering, device, args)

        # Classify patients using Patient Embedding
        if not doClustering:             
            x = self.ClassifyPatients(args)                       
        else:
            x=torch.Tensor([0,0,0]).to(device)

        # Force High entropy in Patient's Embedding.
        # Pat_ent  = [pat_ent2]#loss.Patient_entropy(args,self.S,device)

        # F-test to obtain the least number possible of clusters.
        # f_test_loss = loss.f_test_loss(torch.cat((self.S[0],self.S[1],self.S[2]),dim=1),data.y,device,self.lin1,self.lin2,self.BNLast,args)
        f_test_loss = 0
        
        # Maximize patient class embeddings
        # supconloss = loss.SupConLoss_Total(args,self.S,labels,data,device)
        supconloss = 0
        
        # Unsupervised Clustering
        unsupconloss = loss.UnsupConLoss_Total(args, self.S, data, device, self)
                
        return x, supconloss, minCut2, ortho2, self.s[2:], self.s_interaction, 0, [], pearsonCoeffSUP2, pearsonCoeffUNSUP2, 0, f_test_loss, unsupconloss, pat_ent2, cell_ent_loss2 