import torch
import NaroNet.utils.utilz
import numpy as np
EPS = 1e-15

def Sparse_Pooling(y, x, adj, s,device,args,model):
    
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    out = (torch.matmul(s.transpose(1, 2), x).transpose(1,2).transpose(0,1)/(s.sum(-2)+1e-16)).transpose(0,1).transpose(1,2)  # /(num_nodes/s.shape[-1])
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    # out_adj = out_adj/(out_adj.sum(dim=-1,keepdim=True)+1e-16)        
    # out_adj = out_adj/(out_adj.sum(dim=-2,keepdim=True)+1e-16)

    
    D = out_adj.sum(dim=-1,keepdim=True).pow(-0.5)
    D[D == float('inf')] = 0
    out_adj = out_adj*D
    out_adj = torch.transpose(torch.transpose(out_adj,1,2)*D,1,2)

    minCUT_loss = 0 

    return out, out_adj, minCUT_loss

def Dense_Pooling(y, x, edge_index, s, device, num_nodes,doClustering,args,model,obtainOutX):
        
    
    # Calculate Coarsened Graph
    out_x = torch.zeros(s.shape[0],s.shape[2],args['hiddens'],dtype=torch.float32).to(device)
    out_adj = torch.zeros(s.shape[0],s.shape[2],s.shape[2],dtype=torch.float32).to(device)
    if obtainOutX:
        for i in range(s.shape[0]): 
            # out_x[i,:,:] = torch.matmul(s[i,:num_nodes[i],:].t(),x[i,:num_nodes[i],:])/(num_nodes[i]/s.shape[-1]) if not args['ObjectiveCluster'] else out_x[i,:,:]
            out_x[i,:,:] = (torch.matmul(s[i,:num_nodes[i],:].t(),x[i,:num_nodes[i],:]).t()/(s[i,:num_nodes[i],:].sum(-2)+1e-16)).t()
            out_adj[i,:,:] = torch.matmul(s[i,:num_nodes[i],:].t(),torch.sparse.mm(edge_index[i].float(),s[i,:,:])[:num_nodes[i],:])+1e-16
        
    # Normalize Graph
    if obtainOutX:
        D = out_adj.sum(dim=-1,keepdim=True).pow(-0.5)
        D[D == float('inf')] = 0
        out_adj = out_adj*D
        out_adj = torch.transpose(torch.transpose(out_adj,1,2)*D,1,2)

    # minCUT
    if args['MinCut']:
        minCUT_loss = 0 
        for i in range(s.shape[0]):
            num=torch.trace(out_adj[i,:,:])
            den=torch.trace(torch.matmul(s[i,:,:].t()[:,:num_nodes[i]]*torch.sparse.sum(edge_index[i],1).to_dense()[:num_nodes[i]].float(),s[i,:num_nodes[i],:].float()))
            minCUT_loss = minCUT_loss-(num/(den+1e-16))
        # MultiTask Learning
        minCUT_loss = minCUT_loss/s.shape[0]
        # minCUT_loss=torch.exp(-model.MinCutMTLearning)*(minCUT_loss**2)+model.MinCutMTLearning-1
    else:
        minCUT_loss = 0 

    return out_x, out_adj, minCUT_loss