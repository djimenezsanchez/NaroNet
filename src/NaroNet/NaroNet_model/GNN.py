import torch
import torch.nn.functional as F
from NaroNet.NaroNet_model.torch_geometric_rusty import JumpingKnowledge, uniform

class DenseSAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True):
        super(DenseSAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj   
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        out = torch.matmul(adj, x)
        # print(out.shape)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        # print(adj.sum(dim=-1, keepdim=True).clamp(min=1))
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Dense_SGCConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, khops):
        super(Dense_SGCConv, self).__init__()
        self.khops = khops
        self.in_channels = in_channels
        self.out_channels = out_channels        
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))                
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))        
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)        
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, device, num_nodes=None):
        # Apply Linear layer to the aggregated features
        x = torch.matmul(x, self.weight)                    
        x = x + self.bias 
        # Sparse Multiplication of features and connections
        out1 = torch.zeros(x.shape,dtype=torch.float32).to(device)          
        # out2 = torch.zeros(x.shape,dtype=torch.float32).to(device)          
        # out3 = torch.zeros(x.shape,dtype=torch.float32).to(device)          
        for i in range(out1.shape[0]):                 
            out1[i,:,:] = torch.sparse.mm(edge_index[i],x[i,:,:])        
            # if self.khops==2:
            #     out3[i,:,:] = torch.sparse.mm(edge_index[i],out1[i,:,:])
            # elif self.khops==3:
            # out2[i,:,:] = torch.sparse.mm(edge_index[i],out1[i,:,:])
            #     out3[i,:,:] = torch.sparse.mm(edge_index[i],out2[i,:,:])
            #     x[i,:,:] = torch.sparse.mm(edge_index[i],x[i,:,:]) 
            #     out[i,:,:] = torch.sparse.mm(edge_index[i],x[i,:,:]) 
        # Normalize feature aggregation
        suma = torch.zeros(x.shape[0],x.shape[1],dtype=torch.float32).to(device)        
        for i in range(x.shape[0]):
            suma[i,:] = torch.sparse.sum(edge_index[i],dim=1).to_dense().clamp(min=1)                        
        out1 = out1 / suma.unsqueeze(2)                
        # Apply mask to nodes that are present
        if num_nodes is not None:
            for i in range(x.shape[0]):
                out1[i,num_nodes[i]:,:] = 0
        return out1

class Dense_SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modeltype, normalize=True, bias=True):
        super(Dense_SAGEConv, self).__init__()
        self.modeltype = modeltype
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))        
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)        
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, device, num_nodes=None):
        # Apply Linear layer to the aggregated features
        x = torch.matmul(x, self.weight)                    
        x = x + self.bias 
        # Sparse Multiplication of features and connections
        out = torch.zeros(x.shape,dtype=torch.float32).to(device)  
        for i in range(x.shape[0]):
            out[i,:,:] = torch.sparse.mm(edge_index[i],x[i,:,:]) 
        # Normalize feature aggregation
        suma = torch.zeros(x.shape[0],x.shape[1],dtype=torch.float32).to(device)        
        for i in range(x.shape[0]):
            suma[i,:] = torch.sparse.sum(edge_index[i],dim=1).to_dense().clamp(min=1)                        
        out = out / suma.unsqueeze(2)                
        # Apply mask to nodes that are present
        if num_nodes is not None:
            for i in range(x.shape[0]):
                out[i,num_nodes[i]:,:] = 0
        return out


class Dense_SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, mode='cat'):
        super(Dense_SGC, self).__init__()
        self.in_channels = in_channels
        self.args = args
        # MLP into the features
        self.conv0 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv0BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv0_2 = torch.nn.Linear(hidden_channels, hidden_channels)    
        # Base Graph Neural Network
        self.conv = Dense_SGCConv(hidden_channels, hidden_channels, args['n-hops'])
        self.convBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)                
        # Ending MLP
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv1BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.lin1_2 = torch.nn.Linear(hidden_channels, out_channels)
    def reset_parameters(self):
        # MLP into the features
        self.conv0.reset_parameters()        
        self.conv0_2.reset_parameters()        
        # Base Graph Neural Network
        self.conv.reset_parameters()        
        # Ending MLP      
        self.lin1.reset_parameters()
        self.lin1_2.reset_parameters()
            
    def forward(self, x, edge_index, device, num_nodes,args):        
        x = F.relu(self.conv0(x))
        x = self.conv0BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
        x = F.dropout(x, p=args['dropoutRate'], training=self.training)        
        x = self.conv0_2(x)        
        # k-step Graph Convolution
        x = F.relu(self.conv(x, edge_index, device, num_nodes))        
        x = F.dropout(x, p=args['dropoutRate'], training=self.training)
        x = self.convBN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
        # Ending MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args['dropoutRate'], training=self.training)
        x = self.conv1BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
        x = F.dropout(x, p=args['dropoutRate'], training=self.training)        
        return self.lin1_2(x)  
                
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args,n_hops, mode='SparseMultiplication'):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.args = args
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_hops = n_hops

        # First MLP into the features
        self.conv1MLP = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv1_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    

        # Custom Graph Neural Network
        if args['GraphConvolution']=='ResNet':            
            # Skip Connection for First Graph convolution             
            self.convskip1 = torch.nn.Linear(in_channels, hidden_channels)
            self.convSkip1BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            self.convskip1_1 = torch.nn.Linear(hidden_channels, hidden_channels)    
            # First Graph Convolution
            self.conv1 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                                                          
            self.conv1BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            
            if n_hops==3:
                # Second MLP
                self.conv2MLP = torch.nn.Linear(hidden_channels, hidden_channels)
                self.conv2MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.conv2_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    
                # Skip Connection for Second Graph convolution
                self.convskip2 = torch.nn.Linear(in_channels, hidden_channels)
                self.convSkip2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.convskip2_1 = torch.nn.Linear(hidden_channels, hidden_channels)    
                # Second Graph Convolution
                self.conv2 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                                                          
                self.conv2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)                        
                # Third MLP
                self.conv3MLP = torch.nn.Linear(hidden_channels, hidden_channels)
                self.conv3MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.conv3_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    
                # Skip Connection for Third Graph convolution
                self.convskip3 = torch.nn.Linear(in_channels, hidden_channels)
                self.convSkip3BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.convskip3_1 = torch.nn.Linear(hidden_channels, hidden_channels)   
                # Third Graph Convolution
                self.conv3 = Dense_SAGEConv(hidden_channels, out_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, out_channels)                                                                          
                self.conv3BN = torch.nn.BatchNorm1d(out_channels,track_running_stats=False)                         
            elif n_hops==2:
                # Second MLP
                self.conv2MLP = torch.nn.Linear(hidden_channels, hidden_channels)
                self.conv2MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.conv2_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    
                # Skip Connection for Second Graph convolution             
                self.convskip2 = torch.nn.Linear(in_channels, hidden_channels)
                self.convSkip2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.convskip2_1 = torch.nn.Linear(hidden_channels, hidden_channels)    
                # Second Graph Convolution
                self.conv2 = Dense_SAGEConv(hidden_channels, out_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, out_channels)                                                              
                self.conv2BN = torch.nn.BatchNorm1d(out_channels,track_running_stats=False)  
            elif n_hops==1:
                self.conv1 = Dense_SAGEConv(hidden_channels, out_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, out_channels)                                                                          

        elif args['GraphConvolution']=='IncepNet':
            # MLP To Features         
            self.convskip0 = torch.nn.Linear(in_channels, hidden_channels)
            self.convSkip0BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            self.convskip0_1 = torch.nn.Linear(hidden_channels, hidden_channels)    
            
            # First Graph Convolution        
            self.conv1 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                              
            self.conv1BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            
            # Second Graph Neural Network            
            # MLP for First Graph convolution             
            self.convskip2 = torch.nn.Linear(in_channels, hidden_channels)
            self.convSkip2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            self.convskip2_1 = torch.nn.Linear(hidden_channels, hidden_channels)    
            # First Graph convolution             
            self.conv2 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                              
            self.conv2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)                                                          
            # MLP for Second Graph convolution             
            self.convskip2_3 = torch.nn.Linear(hidden_channels, hidden_channels)
            self.convSkip2_1BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            self.convskip2_4 = torch.nn.Linear(hidden_channels, hidden_channels)    
            # Second Graph convolution             
            self.conv2_2 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                              
            self.conv2_2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)                                                          
                        
            if self.n_hops==3:                
                # MLP for First Graph convolution             
                self.convskip3 = torch.nn.Linear(in_channels, hidden_channels)
                self.convSkip3BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.convskip3_1 = torch.nn.Linear(hidden_channels, hidden_channels)    
                # First Graph convolution             
                self.conv3_1 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                              
                self.conv3BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                # MLP for Second Graph convolution             
                self.convskip3_3 = torch.nn.Linear(hidden_channels, hidden_channels)
                self.convSkip3_1BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.convskip3_4 = torch.nn.Linear(hidden_channels, hidden_channels)    
                # Second Graph convolution             
                self.conv3_2 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                                          
                self.conv3_2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                # MLP for Third Graph convolution             
                self.convskip3_5 = torch.nn.Linear(hidden_channels, hidden_channels)
                self.convSkip3_2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.convskip3_6 = torch.nn.Linear(hidden_channels, hidden_channels)    
                # Third Graph convolution             
                self.conv3_3 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                              
                self.conv3_3BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            # Concatenate Layers
            self.jump = JumpingKnowledge('cat')
            # MLP to join
            if self.n_hops==3:
                self.lin = torch.nn.Linear(hidden_channels + hidden_channels + hidden_channels + hidden_channels, hidden_channels)
            elif self.n_hops==2:
                self.lin = torch.nn.Linear(hidden_channels + hidden_channels + hidden_channels, hidden_channels)        
            elif self.n_hops==1:
                self.lin = torch.nn.Linear(hidden_channels + hidden_channels, hidden_channels)        
            self.linBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        elif args['GraphConvolution']=='JKNet':
            # First Graph Neural Network
            self.conv1 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                                                                      
            self.conv1BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            
            if self.n_hops>=2: 
                # Second MLP
                self.conv2MLP = torch.nn.Linear(hidden_channels, hidden_channels)
                self.conv2MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.conv2_2MLP = torch.nn.Linear(hidden_channels, hidden_channels) 
                # Second Graph Neural Network
                self.conv2 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                                                                      
                self.conv2BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            if self.n_hops==3:   
                # Third MLP
                self.conv3MLP = torch.nn.Linear(hidden_channels, hidden_channels)
                self.conv3MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
                self.conv3_2MLP = torch.nn.Linear(hidden_channels, hidden_channels) 
                # Second Graph Neural Network
                self.conv3 = Dense_SAGEConv(hidden_channels, hidden_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(hidden_channels, hidden_channels)                                                                                          
                self.conv3BN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            # Concatenate Layers
            self.jump = JumpingKnowledge('cat')
            # MLP to join
            if self.n_hops==3:
                self.lin = torch.nn.Linear(hidden_channels + hidden_channels + hidden_channels + hidden_channels, hidden_channels)
            elif self.n_hops==2:
                self.lin = torch.nn.Linear(hidden_channels + hidden_channels + hidden_channels, hidden_channels)
            elif self.n_hops==1:
                self.lin = torch.nn.Linear(hidden_channels + hidden_channels, hidden_channels)
            self.linBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
            self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        # First MLP into the features
        self.conv1MLP.reset_parameters()     
        self.conv1MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv1_2MLP.reset_parameters()                
        # First and Second Graph Neural Network
        self.conv1.reset_parameters()
        self.conv1BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)            
        if self.n_hops==3:
            self.conv2.reset_parameters()
            self.conv2BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)             
        elif self.n_hops==2:
            self.conv2.reset_parameters()
            self.conv2BN = torch.nn.BatchNorm1d(self.out_channels,track_running_stats=False)          
        elif self.n_hops==1:
            self.conv1BN = torch.nn.BatchNorm1d(self.out_channels,track_running_stats=False)            
        # Custom Graph Neural Network
        if self.args['GraphConvolution']=='ResNet':
            # Skip Connection for First Graph convolution             
            self.convskip1.reset_parameters()            
            self.convSkip1BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)        
            self.convskip1_1.reset_parameters()            
            if self.n_hops==3:
                # Third MLP
                self.conv3MLP.reset_parameters()    
                self.conv3MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)           
                self.conv3_2MLP.reset_parameters() 
                # Skip Connection for Third Graph convolution             
                self.convskip3.reset_parameters()         
                self.convSkip3BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)           
                self.convskip3_1.reset_parameters()
                # Third Graph Convolution
                self.conv3BN = torch.nn.BatchNorm1d(self.out_channels,track_running_stats=False)        
                self.conv3.reset_parameters()
            if self.n_hops>=2:
                # Second MLP
                self.conv2MLP.reset_parameters()    
                self.conv2MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)              
                self.conv2_2MLP.reset_parameters() 
                # Skip Connection for Second Graph convolution             
                self.convskip2.reset_parameters()  
                self.convSkip2BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)           
                self.convskip2_1.reset_parameters()


        elif self.args['GraphConvolution']=='IncepNet':            
            # MLP to Features
            self.convskip0.reset_parameters()  
            self.convSkip0BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                  
            self.convskip0_1.reset_parameters()
            self.conv1BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)            

            if self.n_hops>=2:                    
                # Second Graph Convolution
                self.convskip2.reset_parameters()
                self.convSkip2BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)        
                self.convskip2_1.reset_parameters()
                self.convskip2_3.reset_parameters()
                self.convSkip2_1BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)        
                self.convskip2_4.reset_parameters()
                self.conv2_2.reset_parameters()                                    
                self.conv2BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                                                          
                self.conv2_2BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                                                          
            
            if self.n_hops==3:                
                # Third Graph Convolution
                self.convskip3.reset_parameters()
                # MLP for First Graph convolution             
                self.convskip3.reset_parameters()
                self.convskip3_1.reset_parameters()
                # First Graph convolution             
                self.conv3_1.reset_parameters()
                # MLP for Second Graph convolution             
                self.convskip3_3.reset_parameters()
                self.convskip3_4.reset_parameters()
                # Second Graph convolution             
                self.conv3_2.reset_parameters()
                # MLP for Third Graph convolution             
                self.convskip3_5.reset_parameters()
                self.convskip3_6.reset_parameters()
                # Third Graph convolution             
                self.conv3_3.reset_parameters()            
            
            # MLP to join            
            self.lin.reset_parameters()
            self.lin2.reset_parameters()

        elif self.args['GraphConvolution']=='JKNet':            
            
            self.conv1BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)            
            if self.n_hops>=2:
                # Second MLP
                self.conv2MLP.reset_parameters()            
                self.conv2_2MLP.reset_parameters() 
                self.conv2BN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)          
            if self.n_hops==3:
                # Third MLP
                self.conv3MLP.reset_parameters()            
                self.conv3_2MLP.reset_parameters() 
                # Third Graph Convolution
                self.conv3.reset_parameters()

    def MLPintoFeatures(self, x, conv0, conv0_2, conv0BN, args):
        x = F.relu(conv0(x))
        x = conv0BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
        x = F.dropout(x, p=args['dropoutRate'], training=self.training)
        return conv0_2(x)        

    def forward(self, x_raw, edge_index, device, num_nodes,args):        
        if args['GraphConvolution']=='ResNet':            
            # MLP into the features
            x = self.MLPintoFeatures(x_raw, self.conv1MLP, self.conv1_2MLP, self.conv1MLPBN, args)
            # First Graph Convolution + skip Connection             
            x = F.relu(self.conv1(x+self.MLPintoFeatures(x_raw, self.convskip1, self.convskip1_1, self.convSkip1BN, args), edge_index, device, num_nodes))            
            x = F.dropout(x, p=args['dropoutRate'], training=self.training)
            
            
            # Third Graph Convolution
            if self.n_hops>=2:
                x = self.conv1BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
                # MLP + Second Graph Convolution + skip Connection
                x = self.MLPintoFeatures(x, self.conv2MLP, self.conv2_2MLP, self.conv2MLPBN, args)            
                x = F.relu(self.conv2(x + self.MLPintoFeatures(x_raw, self.convskip2, self.convskip2_1, self.convSkip2BN, args), edge_index, device, num_nodes))                        
                x = F.dropout(x, p=args['dropoutRate'], training=self.training) 
                x = self.conv2BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
                
                                
            if self.n_hops==3:
                # MLP + Second Graph Convolution + skip Connection
                x = self.MLPintoFeatures(x, self.conv3MLP, self.conv3_2MLP, self.conv3MLPBN, args)            
                x = F.relu(self.conv3(x + self.MLPintoFeatures(x_raw, self.convskip3, self.convskip3_1, self.convSkip3BN, args), edge_index, device, num_nodes))                        
                x = F.dropout(x, p=args['dropoutRate'], training=self.training) 
                x = self.conv3BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
            
            return x

        elif args['GraphConvolution']=='IncepNet':            
            # First Graph Convolution            
            x1 = self.MLPintoFeatures(x_raw, self.conv1MLP, self.conv1_2MLP, self.conv1MLPBN, args)            
            x1 = F.relu(self.conv1(x1, edge_index, device, num_nodes))            
            x1 = F.dropout(x1, p=args['dropoutRate'], training=self.training)
            x1 = self.conv1BN(x1.transpose(-1,-2)).transpose(-1,-2) if x1.shape[0]>1 else x1

            
            if self.n_hops==1:
                # MLP TO Features
                x_raw = self.MLPintoFeatures(x_raw, self.convskip0, self.convskip0_1, self.convSkip0BN, args)                        
                return self.lin2(self.linBN(F.dropout(F.relu(self.lin(self.jump([x_raw, x1]))), p=args['dropoutRate'], training=self.training).transpose(-1,-2)).transpose(-1,-2)) if x_raw.shape[0]>1 else self.lin2(F.dropout(F.relu(self.lin(self.jump([x_raw, x1]))), p=args['dropoutRate'], training=self.training))      
            
            if self.n_hops>=2:
                # Second Graph Convolution            
                x2 = self.MLPintoFeatures(x_raw, self.convskip2, self.convskip2_1, self.convSkip2BN, args)                        
                x2 = F.relu(self.conv2(x2, edge_index, device, num_nodes))                        
                x2 = F.dropout(x2, p=args['dropoutRate'], training=self.training)            
                x2 = self.conv2BN(x2.transpose(-1,-2)).transpose(-1,-2) if x2.shape[0]>1 else x2
                x2 = self.MLPintoFeatures(x2, self.convskip2_3, self.convskip2_4, self.convSkip2_1BN, args)                                                
                x2 = F.relu(self.conv2_2(x2, edge_index, device, num_nodes))                        
                x2 = F.dropout(x2, p=args['dropoutRate'], training=self.training)
                x2 = self.conv2_2BN(x2.transpose(-1,-2)).transpose(-1,-2) if x2.shape[0]>1 else x2      
                
                if self.n_hops==2: 
                    # MLP TO Features
                    x_raw = self.MLPintoFeatures(x_raw, self.convskip0, self.convskip0_1, self.convSkip0BN, args)                       
                    return self.lin2(self.linBN(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2]))), p=args['dropoutRate'], training=self.training).transpose(-1,-2)).transpose(-1,-2)) if x_raw.shape[0]>1 else self.lin2(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2]))), p=args['dropoutRate'], training=self.training))      

            if self.n_hops==3:
                # Third Graph Convolution            
                x3 = self.MLPintoFeatures(x_raw, self.convskip3, self.convskip3_1, self.convSkip3BN, args)                        
                x3 = F.relu(self.conv3_1(x3, edge_index, device, num_nodes))                        
                x3 = F.dropout(x3, p=args['dropoutRate'], training=self.training)            
                x3 = self.conv3BN(x3.transpose(-1,-2)).transpose(-1,-2) if x3.shape[0]>1 else x3
                x3 = self.MLPintoFeatures(x3, self.convskip3_3, self.convskip3_4, self.convSkip3_1BN, args)                                                
                x3 = F.relu(self.conv3_2(x3, edge_index, device, num_nodes))                        
                x3 = F.dropout(x3, p=args['dropoutRate'], training=self.training)
                x3 = self.conv3_2BN(x3.transpose(-1,-2)).transpose(-1,-2) if x3.shape[0]>1 else x3            
                x3 = self.MLPintoFeatures(x3, self.convskip3_5, self.convskip3_6, self.convSkip3_2BN, args)                                                
                x3 = F.relu(self.conv3_3(x3, edge_index, device, num_nodes))                        
                x3 = F.dropout(x3, p=args['dropoutRate'], training=self.training)
                x3 = self.conv3_3BN(x3.transpose(-1,-2)).transpose(-1,-2) if x3.shape[0]>1 else x3   
                # MLP TO Features
                x_raw = self.MLPintoFeatures(x_raw, self.convskip0, self.convskip0_1, self.convSkip0BN, args)                        
                return self.lin2(self.linBN(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2, x3]))), p=args['dropoutRate'], training=self.training).transpose(-1,-2)).transpose(-1,-2)) if x_raw.shape[0]>1 else self.lin2(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2, x3]))), p=args['dropoutRate'], training=self.training))

                
            
        elif args['GraphConvolution']=='JKNet':
            # MLP into the features
            x_raw = self.MLPintoFeatures(x_raw, self.conv1MLP, self.conv1_2MLP, self.conv1MLPBN, args)
            # First Graph Convolution
            x1 = F.relu(self.conv1(x_raw, edge_index, device, num_nodes))            
            x1 = F.dropout(x1, p=args['dropoutRate'], training=self.training)
            x1 = self.conv1BN(x1.transpose(-1,-2)).transpose(-1,-2) if x1.shape[0]>1 else x1
            if self.n_hops==1:
                return self.lin2(self.linBN(F.dropout(F.relu(self.lin(self.jump([x_raw, x1]))), p=args['dropoutRate'], training=self.training).transpose(-1,-2)).transpose(-1,-2)) if x_raw.shape[0]>1 else self.lin2(F.dropout(F.relu(self.lin(self.jump([x_raw, x1]))), p=args['dropoutRate'], training=self.training))
            if  self.n_hops>=2:
                # MLP + Second Graph Convolution 
                x2 = self.MLPintoFeatures(x1, self.conv2MLP, self.conv2_2MLP, self.conv2MLPBN, args)            
                x2 = F.relu(self.conv2(x2, edge_index, device, num_nodes))                        
                x2 = F.dropout(x2, p=args['dropoutRate'], training=self.training) 
                x2 = self.conv2BN(x2.transpose(-1,-2)).transpose(-1,-2) if x2.shape[0]>1 else x2
                        
            if self.n_hops==3:
                # MLP +Third Graph Convolution
                x3 = F.relu(self.conv3(x2, edge_index, device, num_nodes))                
                x3 = F.dropout(x3, p=args['dropoutRate'], training=self.training)
                x3 = self.conv3BN(x3.transpose(-1,-2)).transpose(-1,-2) if x3.shape[0]>1 else x3
                return self.lin2(self.linBN(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2, x3]))), p=args['dropoutRate'], training=self.training).transpose(-1,-2)).transpose(-1,-2)) if x_raw.shape[0]>1 else self.lin2(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2, x3]))), p=args['dropoutRate'], training=self.training))
            elif self.n_hops==2:
                return self.lin2(self.linBN(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2]))), p=args['dropoutRate'], training=self.training).transpose(-1,-2)).transpose(-1,-2)) if x_raw.shape[0]>1 else self.lin2(F.dropout(F.relu(self.lin(self.jump([x_raw, x1, x2]))), p=args['dropoutRate'], training=self.training))

class GNN_Lin(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args,n_hops, mode='SparseMultiplication'):
        super(GNN_Lin, self).__init__()
        self.in_channels = in_channels
        self.args = args
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_hops = n_hops
        self.mode = mode
                
        # First Graph Convolution
        self.conv1 = Dense_SAGEConv(self.in_channels, self.out_channels, args['modeltype'], normalize=False) if mode=='SparseMultiplication' else Sparse_SAGEConv(self.in_channels, self.out_channels)                                                                          
        
        
    def reset_parameters(self):
        
        self.conv1.reset_parameters()
            
    def forward(self, x_raw, edge_index, device, num_nodes,args):        
        
        x = self.conv1(x_raw, edge_index, device, num_nodes)
                
        if self.mode=='SparseMultiplication':
            # Sparse Multiplication of features and connections
            out = torch.zeros(x.shape,dtype=torch.float32).to(device)  
            for i in range(x.shape[0]):
                out[i,:,:] = torch.sparse.mm(edge_index[i],x[i,:,:]) 
            # Normalize feature aggregation
            suma = torch.zeros(x.shape[0],x.shape[1],dtype=torch.float32).to(device)        
            for i in range(x.shape[0]):
                suma[i,:] = torch.sparse.sum(edge_index[i],dim=1).to_dense().clamp(min=1)                        
            out = out / suma.unsqueeze(2)                
            # Apply mask to nodes that are present
            if num_nodes is not None:
                for i in range(x.shape[0]):
                    out[i,num_nodes[i]:,:] = 0        
            if args['n-hops']==2:
                return x
            else:
                # Sparse Multiplication of features and connections
                x = torch.zeros(x.shape,dtype=torch.float32).to(device)  
                for i in range(x.shape[0]):
                    x[i,:,:] = torch.sparse.mm(edge_index[i],out[i,:,:]) 
                # Normalize feature aggregation
                suma = torch.zeros(x.shape[0],x.shape[1],dtype=torch.float32).to(device)        
                for i in range(x.shape[0]):
                    suma[i,:] = torch.sparse.sum(edge_index[i],dim=1).to_dense().clamp(min=1)                        
                x = x / suma.unsqueeze(2)                
                # Apply mask to nodes that are present
                if num_nodes is not None:
                    for i in range(x.shape[0]):
                        x[i,num_nodes[i]:,:] = 0        
                return x
        elif self.mode!='SparseMultiplication':            
            return x

        

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, args):
        super(LSTM, self).__init__()
        self.in_channels = in_channels
        self.args = args

        # LSTM on features
        self.n_layers = 2
        self.hidden_dim = int(hidden_channels/2)
        self.lstm_layer = torch.nn.LSTM(in_channels, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=args['dropoutRate'])                          

    def forward(self, x_raw, device):                
        # LSTM into the features
        hidden_state = torch.randn(self.n_layers*2, x_raw.shape[0], self.hidden_dim).to(device)
        cell_state = torch.randn(self.n_layers*2, x_raw.shape[0], self.hidden_dim).to(device)
        x, (hn, cn) = self.lstm_layer(x_raw, (hidden_state, cell_state))                
        return x


class phenoNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, mode='SparseMultiplication'):
        super(phenoNN, self).__init__()
        self.in_channels = in_channels
        self.args = args
        self.hidden_channels = hidden_channels

        # # LSTM on features
        # self.n_layers = 2
        # self.hidden_dim = int(hidden_channels/2)
        # self.lstm_layer = torch.nn.LSTM(in_channels, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=args['dropoutRate'])                
        
        # First MLP into the features
        self.conv1MLP = torch.nn.Linear(in_channels, out_channels)
        # self.conv1MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        # self.conv1_2MLP = torch.nn.Linear(hidden_channels, out_channels)    

    def reset_parameters(self):
        # First MLP into the features
        self.conv1MLP.reset_parameters()     
        # self.conv1MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        # self.conv1_2MLP.reset_parameters()                       

    # def MLPintoFeatures(self, x, conv0, conv0_2, conv0BN, args):
    #     x = F.relu(conv0(x))
    #     x = conv0BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
    #     x = F.dropout(x, p=args['dropoutRate'], training=self.training)
    #     return conv0_2(x)        

    def forward(self, x_raw, edge_index, device, num_nodes,args):                       
        return self.conv1MLP(x_raw)

class phenoNN4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, mode='SparseMultiplication'):
        super(phenoNN4, self).__init__()
        self.in_channels = in_channels
        self.args = args
        self.hidden_channels = hidden_channels

        # # LSTM on features
        # self.n_layers = 2
        # self.hidden_dim = int(hidden_channels/2)
        # self.lstm_layer = torch.nn.LSTM(in_channels, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=args['dropoutRate'])                
        
        # First MLP into the features
        self.conv1MLP = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv1_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    

        # Second MLP into the embeddings
        self.conv2MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv2_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)   

        # Third MLP into the embeddings
        self.conv3MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv3_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    

        # Fourth MLP into the embeddings
        self.conv4MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv4MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv4_2MLP = torch.nn.Linear(hidden_channels, out_channels)    

    def reset_parameters(self):
        # First MLP into the features
        self.conv1MLP.reset_parameters()     
        self.conv1MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv1_2MLP.reset_parameters()                       

        # Second MLP into the features
        self.conv2MLP.reset_parameters()     
        self.conv2MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv2_2MLP.reset_parameters()                       

        # Third MLP into the features
        self.conv3MLP.reset_parameters()     
        self.conv3MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv3_2MLP.reset_parameters()     

        # Fourth MLP into the features
        self.conv4MLP.reset_parameters()     
        self.conv4MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv4_2MLP.reset_parameters()     

    def MLPintoFeatures(self, x, conv0, conv0_2, conv0BN, args):
        x = F.relu(conv0(x))
        x = conv0BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
        x = F.dropout(x, p=args['dropoutRate'], training=self.training)
        return conv0_2(x)        

    def forward(self, x_raw, edge_index, device, num_nodes,args):                       
        x = self.MLPintoFeatures(x_raw, self.conv1MLP, self.conv1_2MLP, self.conv1MLPBN, args)
        x = self.MLPintoFeatures(x, self.conv2MLP, self.conv2_2MLP, self.conv2MLPBN, args) + x
        x = self.MLPintoFeatures(x, self.conv3MLP, self.conv3_2MLP, self.conv3MLPBN, args) + x
        return self.MLPintoFeatures(x, self.conv4MLP, self.conv4_2MLP, self.conv4MLPBN, args)

class phenoNN8(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args, mode='SparseMultiplication'):
        super(phenoNN8, self).__init__()
        self.in_channels = in_channels
        self.args = args
        self.hidden_channels = hidden_channels

        # # LSTM on features
        # self.n_layers = 2
        # self.hidden_dim = int(hidden_channels/2)
        # self.lstm_layer = torch.nn.LSTM(in_channels, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=args['dropoutRate'])                
        
        # First MLP into the features
        self.conv1MLP = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv1_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    

        # Second MLP into the embeddings
        self.conv2MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv2_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)   

        # Third MLP into the embeddings
        self.conv3MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv3_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    

        # Fourth MLP into the embeddings
        self.conv4MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv4MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv4_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    
        
        # Fifth MLP into the embeddings
        self.conv5MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv5MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv5_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    
        
        # Sixth MLP into the embeddings
        self.conv6MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv6MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv6_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    
        
        # Seventh MLP into the embeddings
        self.conv7MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv7MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv7_2MLP = torch.nn.Linear(hidden_channels, hidden_channels)    

        # Eighth MLP into the embeddings
        self.conv8MLP = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv8MLPBN = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False)        
        self.conv8_2MLP = torch.nn.Linear(hidden_channels, out_channels) 

    def reset_parameters(self):
        # First MLP into the features
        self.conv1MLP.reset_parameters()     
        self.conv1MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv1_2MLP.reset_parameters()                       

        # Second MLP into the features
        self.conv2MLP.reset_parameters()     
        self.conv2MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv2_2MLP.reset_parameters()                       

        # Third MLP into the features
        self.conv3MLP.reset_parameters()     
        self.conv3MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv3_2MLP.reset_parameters()    

        # Third MLP into the features
        self.conv4MLP.reset_parameters()     
        self.conv4MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv4_2MLP.reset_parameters()    

        # Third MLP into the features
        self.conv5MLP.reset_parameters()     
        self.conv5MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv5_2MLP.reset_parameters()     

        # Third MLP into the features
        self.conv6MLP.reset_parameters()     
        self.conv6MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv6_2MLP.reset_parameters()    
        
        # Third MLP into the features
        self.conv7MLP.reset_parameters()     
        self.conv7MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv7_2MLP.reset_parameters()    

        # Fourth MLP into the features
        self.conv8MLP.reset_parameters()     
        self.conv8MLPBN = torch.nn.BatchNorm1d(self.hidden_channels,track_running_stats=False)                   
        self.conv8_2MLP.reset_parameters()     

    def MLPintoFeatures(self, x, conv0, conv0_2, conv0BN, args):
        x = F.relu(conv0(x))
        if args['Batch_Normalization']:
            x = conv0BN(x.transpose(-1,-2)).transpose(-1,-2) if x.shape[0]>1 else x
        x = F.dropout(x, p=args['dropoutRate'], training=self.training)
        return conv0_2(x)        

    def forward(self, x_raw, edge_index, device, num_nodes,args):                       
        x = self.MLPintoFeatures(x_raw, self.conv1MLP, self.conv1_2MLP, self.conv1MLPBN, args)
        x = self.MLPintoFeatures(x, self.conv2MLP, self.conv2_2MLP, self.conv2MLPBN, args) + x
        x = self.MLPintoFeatures(x, self.conv3MLP, self.conv3_2MLP, self.conv3MLPBN, args) + x
        x = self.MLPintoFeatures(x, self.conv4MLP, self.conv4_2MLP, self.conv4MLPBN, args) + x
        x = self.MLPintoFeatures(x, self.conv5MLP, self.conv5_2MLP, self.conv5MLPBN, args) + x
        x = self.MLPintoFeatures(x, self.conv6MLP, self.conv6_2MLP, self.conv6MLPBN, args) + x
        x = self.MLPintoFeatures(x, self.conv7MLP, self.conv7_2MLP, self.conv7MLPBN, args) + x
        return self.MLPintoFeatures(x, self.conv8MLP, self.conv8_2MLP, self.conv8MLPBN, args)


class Sparse_SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super(Sparse_SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))            
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)        
        uniform(self.in_channels, self.bias)        

    def forward(self, x, adj, device, num_nodes=None):
        
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        # print(out.shape)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        # print(adj.sum(dim=-1, keepdim=True).clamp(min=1))
        out = torch.matmul(out, self.weight)        

        if self.bias is not None:
            out = out + self.bias            

        if self.normalize:
            #outEmb = (outEmb-outEmb.mean(dim=-1,keepdim=True))/outEmb.std(dim=-1,keepdim=True)
            out = F.normalize(out, p=2, dim=-1)            
            #outClust = (outClust-outClust.mean(dim=-1,keepdim=True))/outClust.std(dim=-1,keepdim=True)

        return out

class GCN(torch.nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h

class GloRe_Unit(torch.nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=torch.nn.Conv1d,
                 BatchNormNd=torch.nn.BatchNorm1d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        self.num_in = num_in

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04, track_running_stats=False) # should be zero initialized

    def reset_parameters(self):
        self.conv_state.reset_parameters()
        self.conv_proj.reset_parameters()
        self.gcn.reset_parameters()
        self.conv_extend.reset_parameters()
        self.blocker = torch.nn.BatchNorm1d(self.num_in, eps=1e-04, track_running_stats=False) # should be zero initialized

    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x = x.permute(0, 2, 1)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))
        out = out.permute(0, 2, 1)
        return out
    
    

