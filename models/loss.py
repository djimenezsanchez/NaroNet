import torch
import numpy as np
import torch
import torch.nn.functional as F

def pat_loss(S,args, num_nodes,device):
    # min_Pat_entropy
    if args['Max_Pat_Entropy']:
        pat_ent_loss = ((S/S.sum(-1,keepdim=True)*torch.log(S/S.sum(-1,keepdim=True)+ 1e-15)).sum(dim=1)/torch.log(torch.tensor(S.shape[1],dtype=torch.float32,device=device))).mean()        
        # MultiTask Learning
        # ortho_loss=torch.exp(-model.orthoMTLearning)*(ortho_loss**2)+model.orthoMTLearning-1
    else:
        pat_ent_loss = 0
    return pat_ent_loss

def ortho_and_mincut_loss(data, s, args, num_nodes, device):
    # Orthogonality term
    if args['ortho']:
        ortho_loss = 0
        for i in range(s.shape[0]):
            SS = torch.matmul(s.transpose(-1,-2)[i,:,:num_nodes[i]],s[i,:num_nodes[i],:])
            eye = torch.eye(s.shape[2],device=device)
            num = SS/torch.norm(SS,keepdim=True)
            den = eye/torch.norm(eye,keepdim=True)
            ortho_loss += torch.norm(num-den+1e-15)
        # ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()
        ortho_loss = ortho_loss/s.shape[0]#+ent_loss
        # MultiTask Learning
        # ortho_loss=torch.exp(-model.orthoMTLearning)*(ortho_loss**2)+model.orthoMTLearning-1
    else:
        ortho_loss = 0
    
    # min_Cell_entropy
    if args['min_Cell_entropy']:
        cell_ent_loss =((-s/s.sum(dim=-1,keepdim=True) * torch.log(s/s.sum(dim=-1,keepdim=True) + 1e-15)).sum(dim=-1)/torch.log(torch.tensor(s.shape[2],dtype=torch.float32,device=device))).mean()
        # MultiTask Learning
        # ortho_loss=torch.exp(-model.orthoMTLearning)*(ortho_loss**2)+model.orthoMTLearning-1
    else:
        cell_ent_loss = 0

    # Orthogonality color term
    # if args['orthoColor']:
    #     ortho_color_loss=0
    #     for i in range(s.shape[0]):        
    #         SS = torch.matmul(s[i,:,:].t(),data.x[i,:data.num_nodes[i],:])
    #         SS = torch.matmul(SS,SS.t())
    #         eye = torch.eye(s.shape[2],device=device)
    #         num = SS/torch.norm(SS,keepdim=True)
    #         den = eye/torch.norm(eye,keepdim=True)            
    #         ortho_color_loss += torch.norm(num-den+1e-15)  
    #     # MultiTask Learning
    #     ortho_color_loss=ortho_color_loss/s.shape[0]
    #     # ortho_color_loss=torch.exp(-model.orthoColorMTLearning)*(ortho_color_loss**2)+model.orthoColorMTLearning-1
    # else:
    #     ortho_color_loss=torch.zeros(1,device=device).squeeze(dim=0)

    if args['orthoColor']:
        ortho_color_loss=0
        if hasattr(data.num_nodes, '__iter__'):
            for i in range(s.shape[0]): 
                SS = torch.matmul(s[i,:data.num_nodes[i],:].t(),data.x[i,:data.num_nodes[i],:])
                unsupconloss1, unsup_acc1 = unsup_ColorLoss(SS,data.y,SS,device,temperature=0.07)
                ortho_color_loss += unsupconloss1
        ortho_color_loss = ortho_color_loss/s.shape[0]
    else:
        ortho_color_loss=torch.zeros(1,device=device).squeeze(dim=0)

    # Pearson coefficient Using the labels.
    pearsonCoeffSUP = 0
    if args['pearsonCoeffSUP']:
        ss = (s.sum(1)/s.shape[1])
        for feature in range(s.shape[-1]):
            pearsonCoeffSUP = pearsonCoeffSUP + torch.pow(utilz.pearson_corr(ss[:,feature],y[:,0]),2)
        pearsonCoeffSUP=-pearsonCoeffSUP.sqrt()/s.shape[-1]
        # pearsonCoeffSUP=torch.exp(-model.pearsonSupMTLearning)*(pearsonCoeffSUP**2)+model.pearsonSupMTLearning-1

    # Pearson coefficient of the matrix
    pearsonCoeffUnsup=0
    if args['pearsonCoeffUNSUP']:
        ss = torch.zeros(s.shape[0],s.shape[-1],dtype=torch.float32).to(device)        
        for i in range(s.shape[0]):
            ss[i,:] = (s[i,:num_nodes[i],:].sum(0)/num_nodes[i])
        pearsonCoeffUnsup = utilz.cov_matrix(ss, rowvar=True)
        pearsonCoeffUnsup = -pearsonCoeffUnsup.pow(2).sum().sqrt()
        # pearsonCoeffUnsup=torch.exp(-model.pearsonUnsupMTLearning)*(pearsonCoeffUnsup**2)+model.pearsonUnsupMTLearning-1
    
    return ortho_color_loss, pearsonCoeffSUP, pearsonCoeffUnsup, ortho_loss, cell_ent_loss
    
def Patient_entropy(args,S,device):
    # Keep High Entropy in the network
    if args['Max_Pat_Entropy']:                                          
        NN_loss1 = ((S[0]/S[0].sum(-1,keepdim=True)*torch.log(S[0]/S[0].sum(-1,keepdim=True)+ 1e-15)).sum(dim=1)/torch.log(torch.tensor(S[0].shape[1],dtype=torch.float32,device=device))).mean()
        NN_loss2 = ((S[1]/S[1].sum(-1,keepdim=True)*torch.log(S[1]/S[1].sum(-1,keepdim=True)+ 1e-15)).sum(dim=1)/torch.log(torch.tensor(S[1].shape[1],dtype=torch.float32,device=device))).mean()
        NN_loss3 = ((S[2]/S[2].sum(-1,keepdim=True)*torch.log(S[2]/S[2].sum(-1,keepdim=True)+ 1e-15)).sum(dim=1)/torch.log(torch.tensor(S[2].shape[1],dtype=torch.float32,device=device))).mean()
        print('Entropy Phenotypes: ',NN_loss1.item(),' Entropy TissueCommunities: ',NN_loss2.item(), 'Entropy TissueCommunitiesInter: ',NN_loss3.item(),'Min_Clust: ',S[0].sum(-2).min().item())
        return NN_loss1, NN_loss2, NN_loss3
    else:
        return 0, 0, 0

def NearestNeighborClassification(self):
    NN_loss, _, _, _ = utilz.nearestNeighbor_loss(self.training, args, self.sPatient[-1], data.y.long(), self.NNCPosition,index,Indices,labels,device,self)                                
    NN_loss = -(-torch.cat((self.S[0],self.S[1],self.S[2]),dim=1) * torch.log(torch.cat((self.S[0],self.S[1],self.S[2]),dim=1) + 1e-15)).sum(dim=-1).mean()                

def f_test_loss(features,labels,device,lin1,lin2,BN,args):
    
    if args['F-test']:
        # Unrestricted Model
        UM = F.relu(lin1(features)) # torch.cat(self.X,dim=-1)            
        UM = lin2(UM)         
        # Model Loss of Unrestricted Model
        unrestricted_loss = F.cross_entropy(UM, labels[:,0].long())     
        pred = UM.max(1)[1].detach().cpu().numpy()
        print('train_acc:',np.equal(pred,labels[:,0].cpu().numpy()).mean())

        # # Sum of Squares Residuals of unrestricted model
        # unrestricted_RSS = torch.pow((torch.matmul(features,regression_model.weight.t())+regression_model.bias).squeeze()-labels,2).sum()    
        # # Leave-one-out f-test model
        restricted_loss = torch.zeros(features.shape[1],device=device)
        
        # For copies
        # W_f = regression_model.weight.clone()
        indices = torch.arange(features.shape[1],device=device)
        
        # Restrict model per feature.
        for f in range(features.shape[1]):
            # Put Weight of f_index to zero        
            # regression_model.weight[0,f] = 0

            # Obtain Sum of Squares of restricted model        
            Restricted_Info = torch.zeros(features.shape,device=device)        
            Restricted_Info[:,indices!=f] = features[:,indices!=f]
            restricted_loss[f] = F.cross_entropy(lin2(BN(F.relu(lin1(Restricted_Info)))), labels[:,0].long()) if Restricted_Info.shape[0]>1 else F.cross_entropy(lin2(F.relu(lin1(Restricted_Info))), labels[:,0].long())        
            # torch.cat(self.X,dim=-1)                 
            
            
            # restricted_RSS = torch.pow((torch.matmul(features[:,x[x!=f]],regression_model.weight[:,x[x!=f]].t())+regression_model.bias).squeeze()-labels,2).sum()

            # F-test
            # loss[f] = (restricted_RSS-unrestricted_RSS)/(features.shape[0]-features.shape[1])

        # loss = F.relu(loss)
        # Sparsify the unrestricted models
        restricted_loss = (restricted_loss/unrestricted_loss)  
        restricted_loss = restricted_loss/restricted_loss.sum()  
        restricted_loss = -(restricted_loss * torch.log(restricted_loss + 1e-15)).mean()
        # restricted_loss = -restricted_loss.max()
        # restricted_loss = -(torch.sqrt(restricted_loss.shape[0]*torch.ones(1,device=device))-(restricted_loss.sum()/(torch.pow(restricted_loss,2).sum()+1e-16)))/(torch.sqrt(restricted_loss.shape[0]*torch.ones(1,device=device))-1)
        print('sparsify the f-test',restricted_loss, 'loss of original Model',unrestricted_loss)
        f_test_loss = unrestricted_loss+restricted_loss
    else:
        f_test_loss = 0

    return f_test_loss

def SupConLoss(features,labels,device,temperature=0.07):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    # device = (torch.device('cuda')
    #           if features.is_cuda
    #           else torch.device('cpu'))
    
    loss = torch.zeros(1,device=device)
    features = features.unsqueeze(dim=1)
    Contrast_acc = 0
    n_iter = 0

    # Calculate loss only if there are repeated values in loss
    a, rep = np.unique(labels.cpu().numpy(),return_counts=True)
    if any(rep>1):
        # Iterate Over each patient i.        
        for i in range(features.shape[0]): 
            
            # Reinitialize patient_i_loss
            patient_i_loss = torch.zeros(1,device=device)

            # Iterate Over each patient j.
            for j in range(features.shape[0]): 
                
                # Select patient if they have same labels.
                if (i!=j) and (labels[i]==labels[j]):                 
                    # Apply supContrast Loss
                    patient_i_loss += torch.log(
                        torch.exp(F.cosine_similarity(features[i],features[j])/temperature)/
                        torch.exp(F.cosine_similarity(features[i],features[labels[i]!=labels].squeeze(dim=1))/temperature).mean())
                    Contrast_acc += F.cosine_similarity(features[i],features[j])-F.cosine_similarity(features[i],features[labels[i]!=labels].squeeze(dim=1)).mean()
                    n_iter += 1 

            # Normalize patient_i_loss
            loss += (-torch.ones(1,device=device)/((labels[i]==labels).sum()-1+1e-16))*patient_i_loss
        
        loss /= features.shape[0]
                    
    return loss, Contrast_acc/(n_iter+1e-15)

def SupConLoss_Total(args,S,labels,data,device):
    if args['orthoColor']:
        # supconloss = torch.relu(self.SUPCONlin(torch.cat((self.S[0],self.S[1],self.S[2]),dim=1)))
        # supconloss = utilz.SupConLoss(torch.cat((self.S[0],self.S[1],self.S[2]),dim=1),data.y,device,temperature=0.07)
        if args['F-test']:
            supconloss = SupConLoss(S[0][:,:max(labels)+1],data.y,device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[0][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
            supconloss += SupConLoss(S[1][:,:max(labels)+1],data.y,device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[1][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
            supconloss += SupConLoss(S[2][:,:max(labels)+1],data.y,device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[2][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
            # f_test_loss = -self.S[0][:,max(labels)+1:].mean()
            # f_test_loss += -self.S[1][:,max(labels)+1:].mean()
            # f_test_loss += -self.S[2][:,max(labels)+1:].mean()
            # f_test_loss = f_test_loss*0.001
        else:
            supconloss1, sup_acc1  = SupConLoss(S[0],data.y[:,0],device,temperature=0.07)
            supconloss2, sup_acc2 = SupConLoss(S[1],data.y[:,0],device,temperature=0.07)
            supconloss3, sup_acc3 = SupConLoss(S[2],data.y[:,0],device,temperature=0.07)
            if np.argmin([sup_acc1,sup_acc2,sup_acc3])==0:
                supconloss=supconloss1
            elif np.argmin([sup_acc1,sup_acc2,sup_acc3])==1:
                supconloss=supconloss2
            else:
                supconloss=supconloss3
            print('Supervised_Contrast_accuracy: ',(sup_acc1).item(),+sup_acc2.item(), sup_acc3.item())
    else:
        supconloss=0

    return supconloss

def unsup_ColorLoss(projection,labels, features,device,temperature=0.07):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    # device = (torch.device('cuda')
    #           if features.is_cuda
    #           else torch.device('cpu'))
    
    loss = torch.zeros(1,device=device)
    projection = projection/projection.max()
    projection = projection.unsqueeze(dim=1)
    features = features/features.max()
    features = features.unsqueeze(dim=1)

    unsup_Contrast_acc = 0 
    n_iter = 0

    # Calculate loss only if there are more than One Image.
    if labels.shape[0]>2:
        # Iterate Over each patient i.        
        for i in list(range(0,projection.shape[0],1)): 
            
            # Reinitialize patient_i_loss
            patient_i_loss = torch.zeros(1,device=device)

            # Iterate Over each patient j.
            for j in range(projection.shape[0]): 
                
                # Select patient if they have same labels.
                if (i!=j):                 
                    # Apply supContrast Loss
                    patient_i_loss += torch.exp(F.cosine_similarity(projection[i],projection[j]))
                    # unsup_Contrast_acc += F.cosine_similarity(features[i],features[i+1])-F.cosine_similarity(features[i],features[j])
                    n_iter += 1 

            # Normalize patient_i_loss
            loss += patient_i_loss
        
        loss /= projection.shape[0]*(projection.shape[0]-1)
                    
    return loss, unsup_Contrast_acc/(n_iter+1e-16)

def UnsupConLoss(projection,labels, features,device,temperature=0.07):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    # device = (torch.device('cuda')
    #           if features.is_cuda
    #           else torch.device('cpu'))
    
    loss = torch.zeros(1,device=device)

    projection = projection.unsqueeze(dim=1)
    features = features.unsqueeze(dim=1)

    unsup_Contrast_acc = 0 
    n_iter = 0

    # Calculate loss only if there are more than One Image.
    if labels.shape[0]>2:
        # Iterate Over each patient i.        
        for i in list(range(0,projection.shape[0],2)): 
            
            # Reinitialize patient_i_loss
            patient_i_loss = torch.zeros(1,device=device)

            # Iterate Over each patient j.
            for j in range(projection.shape[0]): 
                
                # Select patient if they have same labels.
                if (i!=j) and ((i+1)!=j):                 
                    # Apply supContrast Loss
                    patient_i_loss += -torch.log(
                        torch.exp(F.cosine_similarity(projection[i],projection[i+1])/temperature)/
                        torch.exp(F.cosine_similarity(projection[i],projection[j])/temperature).mean())
                    unsup_Contrast_acc += F.cosine_similarity(features[i],features[i+1])-F.cosine_similarity(features[i],features[j])
                    n_iter += 1 

            # Normalize patient_i_loss
            loss += patient_i_loss
        
        loss /= projection.shape[0]*(projection.shape[0]-2)
                    
    return loss, unsup_Contrast_acc/(n_iter+1e-16)

def UnsupConLoss_Total(args, S, data, device, self):
    if args['UnsupContrast'] and self.training:
        # unsupconloss1, unsup_acc1 = utilz.UnsupConLoss(self.lin1_unsupB(F.relu(self.lin1_unsupA(self.S[0]))),data.y,self.S[0],device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[0][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
        # unsupconloss2, unsup_acc2 = utilz.UnsupConLoss(self.lin2_unsupB(F.relu(self.lin2_unsupA(self.S[1]))),data.y, self.S[1],device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[1][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
        # unsupconloss3, unsup_acc3 = utilz.UnsupConLoss(self.lin3_unsupB(F.relu(self.lin3_unsupA(self.S[2]))),data.y,self.S[2],device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[2][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
        unsupconloss1, unsup_acc1 = UnsupConLoss(S[0],data.y,S[0],device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[0][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
        unsupconloss2, unsup_acc2 = UnsupConLoss(S[1],data.y,S[1],device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[1][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                
        unsupconloss3, unsup_acc3 = UnsupConLoss(S[2],data.y,S[2],device,temperature=0.07)# + torch.stack([-(i * torch.log(i + 1e-15)).sum() for i in self.S[2][:,:max(labels)+1]/self.S[0][:,:max(labels)+1].sum(-1).unsqueeze(-1)]).mean()                    
        # if np.argmin([unsup_acc1,unsup_acc2,unsup_acc3])==0:
        #     unsupconloss=unsupconloss1
        # elif np.argmin([unsup_acc1,unsup_acc2,unsup_acc3])==1:
        #     unsupconloss=unsupconloss2
        # else:
        #     unsupconloss=unsupconloss2
        unsup_acc = (unsup_acc1+unsup_acc2+unsup_acc3)/3
        # else:
        #     unsupconloss=unsupconloss3
        # print('Unsupervised_accuracy: ',(unsup_acc1).item(),+unsup_acc2.item(), unsup_acc3.item())
        print('Unsupervised_accuracy: ',unsup_acc)
        return unsupconloss1, unsup_acc1, unsupconloss2, unsup_acc2, unsupconloss3, unsup_acc3 
    else: 
        return 0,0,0,0,0,0

def Lasso_Feat_Selection(layer1,layer2):
    return 0.1*layer1.weight.abs().sum() + 0.1*layer2.weight.abs().sum()


    

