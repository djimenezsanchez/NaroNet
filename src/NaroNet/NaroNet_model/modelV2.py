import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import JumpingKnowledge
import models.pooling as pooling
import models.GNN as GNN
import utilz

class DiffPool_Dense_ATT_Pheno(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, num_nodes, clusts, args):
        super(DiffPool_Dense_ATT_Pheno, self).__init__()
        # Initialization
        self.features = num_features
        self.args = args
        num_nodes = clusts[0]
        self.hidden = hidden

        # # LSTM on features
        # if args['LSTMonFeatures']:
        #     n_layers = 6
        #     lstm_layer = nn.LSTM(num_features, floor(hidden/2), 6, batch_first=True)        
        #     hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
        #     cell_state = torch.randn(n_layers, batch_size, hidden_dim)

        # Attention and Readout
        if args['isAttentionLayer']:
            self.attention1 = torch.nn.Linear(hidden,hidden,bias=True)
            self.BatchNormATT = torch.nn.BatchNorm1d(hidden,track_running_stats=False)
            self.attention2 = torch.nn.Linear(hidden,1,bias=True)
        else:
            self.attention1, self.BatchNormATT, self.attention2 = [], [], []
        if args['ReadoutFunction']=='DeepSets' and not args['ObjectiveCluster']:
            self.deepsets1=torch.nn.Linear(hidden,hidden+hidden,bias=True)
            self.deepsets2=torch.nn.Linear(num_nodes,hidden,bias=True)
        else:
            self.deepsets1, self.deepsets2 = [], []

        # Graph Neural Network Embedding
        if args['modeltype']=='SAGE':
            self.embed_pool_block_emb = GNN.GNN(num_features, hidden, hidden, args, mode='SparseMultiplication') if not args['ObjectiveCluster'] else []
            self.embed_pool_block_clust = GNN.GNN(num_features, hidden,  num_nodes, args, mode='SparseMultiplication')
        elif args['modeltype']=='SGC':
            self.embed_pool_block_emb = GNN.Dense_SGC(num_features, hidden, hidden, args) if not args['ObjectiveCluster'] else []
            self.embed_pool_block_clust = GNN.Dense_SGC(num_features, hidden,  num_nodes, args)        
        self.BatchNorm = torch.nn.BatchNorm1d(hidden,track_running_stats=False)
        self.embed_blocks_emb = torch.nn.ModuleList()
        self.embed_blocks_clust = torch.nn.ModuleList()
        self.attention1_blocks = torch.nn.ModuleList()        
        self.attention2_blocks = torch.nn.ModuleList()
        self.BatchNormATT2 = torch.nn.ModuleList()         
        self.deepsets1_blocks=torch.nn.ModuleList()
        self.deepsets2_blocks=torch.nn.ModuleList()

        num_total_clusts = num_nodes
        for i in range(1,len(clusts)):
            numtotalNodes = num_nodes
            num_nodes = clusts[i]
            num_total_clusts += num_nodes
            self.embed_blocks_emb.append(GNN.GNN(hidden, hidden, hidden, args, mode='Multiplication'))
            self.embed_blocks_clust.append(GNN.GNN(hidden, hidden, num_nodes, args, mode='Multiplication'))
            self.attention1_blocks.append(torch.nn.Linear(hidden,hidden,bias=True)) if args['isAttentionLayer'] else self.attention1_blocks.append(torch.nn.Linear(1,1,bias=True))
            self.attention2_blocks.append(torch.nn.Linear(hidden,1,bias=True)) if args['isAttentionLayer'] else self.attention2_blocks.append(torch.nn.Linear(1,1,bias=True))
            self.BatchNormATT2.append(torch.nn.BatchNorm1d(hidden,track_running_stats=False)) if args['isAttentionLayer'] else self.BatchNormATT2.append(torch.nn.Linear(1,1,bias=True))
            if args['ReadoutFunction']=='DeepSets':
                self.deepsets1_blocks.append(torch.nn.Linear(hidden,hidden+hidden,bias=True))
                self.deepsets2_blocks.append(torch.nn.Linear(num_nodes,hidden,bias=True))    
            else:
                self.deepsets1_blocks.append(torch.nn.Linear(1,1,bias=True))
                self.deepsets2_blocks.append(torch.nn.Linear(1,1,bias=True))     

        # Ending layers
        if args['NearestNeighborClassification']:
            self.lin1NNC = torch.nn.Linear(sum([c+(c*(c-1)) for c in clusts]), 10, bias=True)
            self.BNNNC = torch.nn.BatchNorm1d(hidden+hidden,track_running_stats=False)
            self.lin2NNC = torch.nn.Linear(hidden+hidden, 10, bias=True)
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = torch.nn.Linear((hidden)*(len(clusts)), hidden+hidden, bias=True) if not args['ObjectiveCluster'] else torch.nn.Linear(sum([c for c in clusts]), hidden+hidden, bias=True) # torch.nn.Linear(sum([c+(c*(c-1)) for c in clusts]), hidden+hidden, bias=True)
        self.BNLast = torch.nn.BatchNorm1d(hidden+hidden,track_running_stats=False)
        self.lin2 = torch.nn.Linear(hidden+hidden, num_classes, bias=True)

        # # MultiTask Learning
        # if args['pearsonCoeffSUP']:
        #     self.pearsonSupMTLearning=torch.zeros((1)).cuda()#torch.nn.Parameter(torch.zeros((1)))
        # if args['pearsonCoeffUNSUP']:
        #     self.pearsonUnsupMTLearning=torch.zeros((1)).cuda()#torch.nn.Parameter(torch.zeros((1)))
        # if args['orthoColor']:
        #     self.orthoColorMTLearning=torch.zeros((1)).cuda()#torch.nn.Parameter(torch.zeros((1)))
        # if args['ortho']:
        #     self.orthoMTLearning=torch.zeros((1)).cuda()#torch.nn.Parameter(torch.zeros((1)))
        # if args['MinCut']:
        #     self.MinCutMTLearning=torch.zeros((1)).cuda()#torch.nn.Parameter(torch.zeros((1)))
        # if args['NearestNeighborClassification']:
        #     self.NNClsfctionMTLearning=torch.zeros((1)).cuda()#torch.nn.Parameter(torch.zeros((1)))
        # self.NNClsfctionMTLearning=torch.zeros((1)).cuda()#torch.nn.Parameter(torch.zeros((1)))
        

    def reset_parameters(self):        
        try:
            self.embed_pool_block_emb.reset_parameters()      
        except:
            self.embed_pool_block_emb=[]
        self.embed_pool_block_clust.reset_parameters()              
        try:
            self.attention1.reset_parameters() 
            self.attention2.reset_parameters()        
        except:
            self.attention1, self.attention2 = [], []
        try: 
            self.deepsets1.reset_parameters()
            self.deepsets2.reset_parameters()
            for deepsets1, deepsets2 in zip(self.deepsets1_blocks, self.deepsets2_blocks):
                deepsets1.reset_parameters()
                deepsets2.reset_parameters()
        except:
            self.deepsets1, self.deepsets2 = [], []
        for block_emb, block_clust, attention1, attention2 in zip(self.embed_blocks_emb,self.embed_blocks_clust,self.attention1_blocks, self.attention2_blocks):
            block_emb.reset_parameters()
            block_clust.reset_parameters()
            attention1.reset_parameters()
            attention2.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        try:
            self.lin1NNC.reset_parameters()
            self.lin2NNC.reset_parameters()
        except:
            self.lin1NNC, self.lin2NNC= [], []        
            

    def attentionLayer(self,x,edge_index,attention1,BatchNormATT,attention2,deepsets1,deepsets2,args,device,attentionVectList,s_interaction):
        if args['isAttentionLayer']:                
            attentionVect = BatchNormATT(F.dropout(F.relu(attention1(x)),p=args['dropoutRate'], training=self.training).transpose(-1,-2)).transpose(-1,-2)                
            attentionVect = BatchNormATT(attentionVect.transpose(-1,-2)).transpose(-1,-2) if attentionVect.shape[0]>1 else attentionVect
            attentionVect = attention2(attentionVect).squeeze(-1)
            attentionVect = F.softmax(attentionVect,dim=-1)
            attentionVect = [torch.where(attentionVect>args['attntnThreshold']/attentionVect.shape[1], attentionVect, torch.tensor([0],dtype=torch.float32).to(device))]                
            x = torch.transpose(torch.transpose(torch.transpose(torch.transpose(x,1,0),0,2)*(attentionVect[-1]),0,2),1,0)
            s_interaction.append(edge_index)
            edge_index = torch.transpose(torch.transpose(torch.transpose(edge_index,1,0)*(attentionVect[-1]),2,0)*(attentionVect[-1]),1,0)
            xs=x.sum(dim=1) if args['ReadoutFunction']=='SUM' else x.max(dim=1)[0]/torch.nonzero(attentionVect[-1]).size(1) if args['ReadoutFunction']=='MAX' else x.sum(dim=1)/torch.nonzero(attentionVect[-1]).size(1)
            attentionVectList.append(attentionVect)
        else:
            s_interaction.append(edge_index)
            xs=x.sum(dim=1) if args['ReadoutFunction']=='SUM' else x.max(dim=1)[0] if args['ReadoutFunction']=='MAX' else deepsets2(deepsets1(x).sum(dim=-1))
            attentionVectList=[]
        return s_interaction, edge_index, x, xs, attentionVectList

    def forward(self, data, device, saveClusters, trainClustering, doClustering,index,Indices,labels,args):

        # First model instance to obtain the interaction-graph
        if doClustering:
            # GNN
            self.s = self.embed_pool_block_clust(data.x, data.edge_index, device, data.num_nodes,args)    
            data.x = [] if args['ObjectiveCluster'] else self.embed_pool_block_emb(data.x, data.edge_index, device, data.num_nodes,args)    
            
            # Apply softmax to Cluster Assignment
            p=0.0
            self.s=F.softmax(self.s,dim=-1)
            self.s = torch.where(self.s>args['attntnThreshold']/self.s.shape[2], self.s, torch.tensor([0],dtype=torch.float32).to(device))
            self.s = [self.s/self.s.sum(dim=-1,keepdim=True)]
            # print(sum(torch.bincount(s[-1][0,:data.num_nodes[0],:].argmax(-1))>0))
            
            # Node-Pooling
            data.x, data.edge_index, ortho_color, pearsonCoeffSUP, pearsonCoeffUnsup, minCut, ortho = pooling.Dense_Pooling(data.y,data.x, data.edge_index, self.s[-1],self.BatchNorm, device, data.num_nodes,doClustering, args,self)  # Regions                        
            # Save Patient-Matrix for later on.            
            if args['ObjectiveCluster'] or args['NearestNeighborClassification']:
                self.sPatient = [torch.zeros(self.s[0].shape[0],self.s[0].shape[-1],dtype=torch.float32).to(device)]
                for i in range(self.s[0].shape[0]):
                    self.sPatient[0][i,:] = (self.s[0][i,:data.num_nodes[i],:].sum(0)/data.num_nodes[i])            
            return data,ortho_color, minCut, ortho, pearsonCoeffSUP, pearsonCoeffUnsup, self.s[-1]            
        else:
            # Use attention in our neural netwwork or not, and apply readout Function to X
            s_interaction, data.edge_index, data.x, ReadoutX, attentionVect = self.attentionLayer(data.x, data.edge_index,self.attention1,self.BatchNormATT,self.attention2, self.deepsets1, self.deepsets2,args,device,[],[])            
            xs = [ReadoutX]            
            minCut=0
            ortho=0
            ortho_color=0
            pearsonCoeffSUP=0
            pearsonCoeffUNSUP=0
            self.edge_index = [data.edge_index]
        for embed, clust, attention1, attention2, batchnorm, deepsets1, deepsets2 in zip(self.embed_blocks_emb, self.embed_blocks_clust, self.attention1_blocks,self.attention2_blocks, self.BatchNormATT2,self.deepsets1_blocks,self.deepsets2_blocks):                                           
            data.x = embed(data.x, data.edge_index,device, None,args) # Feats  
            sux = clust(data.x, data.edge_index, device, None,args) # Clust  
            s_aux = F.softmax(sux,dim=-1)
            s_aux=torch.where(s_aux>args['attntnThreshold']/s_aux.shape[2],s_aux, torch.tensor([0],dtype=torch.float32).to(device))
            s_aux = s_aux/s_aux.sum(dim=-1,keepdim=True)                      
            data.x, data.edge_index, min_next, ortho_next, ortho_color_next, pearsonCoeffSUP_next, pearsonCoeffUNSUP_next = pooling.Sparse_Pooling(data.y,data.x, data.edge_index, s_aux,device,args,self)                        
            self.s.append(s_aux)
            self.edge_index.append(data.edge_index) if args['ObjectiveCluster'] else []
            # Save Patient-Matrix for later on.
            if args['ObjectiveCluster'] or args['NearestNeighborClassification']:
                self.sPatient.append(torch.zeros(s_aux.shape[0],s_aux.shape[-1],dtype=torch.float32).to(device))
                for i in range(self.sPatient[-1].shape[0]):
                    self.sPatient[-1][i,:] = (s_aux[i,:,:].sum(0)/s_aux[i,:,:].shape[0])
            
            # Attention layer or not            
            if not doClustering:
                # Use attention in our neural network or not, and apply readout Function to X                                                                                                                
                s_interaction, data.edge_index, data.x, ReadoutX, attentionVect = self.attentionLayer(data.x,data.edge_index,attention1,batchnorm,attention2,deepsets1,deepsets2,args,device,attentionVect,s_interaction)                
                xs.append(ReadoutX)
                
            # Add unsupevised losses
            minCut += min_next
            ortho += ortho_next
            ortho_color += ortho_color_next
            pearsonCoeffSUP += pearsonCoeffSUP_next
            pearsonCoeffUNSUP += pearsonCoeffUNSUP_next
        # Normalize Losses
        minCut /= len(self.s)-1 if len(self.s)>1 else 1
        ortho /= len(self.s)-1 if len(self.s)>1 else 1
        ortho_color /=len(self.s)-1 if len(self.s)>1 else 1
        pearsonCoeffUNSUP /= len(self.s)-1 if len(self.s)>1 else 1
        pearsonCoeffSUP /= len(self.s)-1 if len(self.s)>1 else 1
        attention_sparseness = 0
        
        # Final Layer
        if not doClustering:            
            if args['ObjectiveCluster']: # Just use S to classify.
                # x=  F.relu(self.lin1(self.jump(self.sPatient))) 
                # x = self.BNLast(x) if x.shape[0]>1 else x
                # x = F.dropout(x, p=args['dropoutRate'], training=self.training)
                # x = self.lin2(x) 
                x = F.relu(self.lin1(torch.cat(self.sPatient,1)))
                x = self.BNLast(x) if x.shape[0]>1 else x
                x = F.dropout(x, p=args['dropoutRate'], training=self.training)
                x = self.lin2(x) 
            else: # Use X to classify
                x = self.jump(xs)
                x = F.relu(self.lin1(x))
                x = self.BNLast(x) if x.shape[0]>1 else x
                x = F.dropout(x, p=args['dropoutRate'], training=self.training)
                x = self.lin2(x)
            if args['NearestNeighborClassification']:                  
                # self.edge_index = [e[torch.nonzero((torch.eye(e.shape[-1],device=device)-1).abs().expand_as(e),as_tuple=True)].view(-1,(e.shape[-1]-1)*e.shape[-1]) for e in self.edge_index] # Extract edges from Adjacency Matrix.
                # self.NNx = torch.cat((self.jump(self.sPatient),self.jump(self.edge_index)),dim=1)
                # self.NNx = self.lin1NNC(self.NNx)
                # self.NNx = self.BNNNC(self.NNx) if self.NNx.shape[0]>1 else self.NNx
                # self.NNx = self.lin2NNC(self.NNx)                
                NN_loss, _, _, _ = utilz.nearestNeighbor_loss(self.training, args, self.sPatient[-1], data.y.long(), self.NNCPosition,index,Indices,labels,device,self)                
                # self.NNx = self.jump(self.s) 
                # 
            else:
                NN_loss=0                            
        else:
            x=torch.Tensor([0,0,0]).to(device)
        return x, ortho_color, minCut, ortho, self.s[1:], s_interaction, attention_sparseness, attentionVect, pearsonCoeffSUP, pearsonCoeffUNSUP, NN_loss
