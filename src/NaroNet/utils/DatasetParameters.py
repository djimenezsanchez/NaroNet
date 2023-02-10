
import argparse
import numpy as np
from hyperopt import hp

def parameters(path, debug):

    # args=DefaultParameters(path)
    args={}
    args['path'] = path

    # Use optimized parameters depending on the experiment.
    if 'Cytof52Breast' in path:
        # Optimization Parameters
        args['epochs'] = 15 if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.1 if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 10 if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)
        args['weight_decay'] = 0.01 if debug else hp.uniform('weight_decay', 0.01, 0.1)
        args['batch_size'] = 6 if debug else hp.quniform('batch_size', 4, 20, 1)
        args['lr'] = 0.001 if debug else hp.uniform('lr', 0.0001, 0.1)
        args['useOptimizer'] = 0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        args['context_size'] = 1 if debug else hp.choice("context_size", [10, 30, 50])
        args['num_classes'] = 2
        args['hiddens'] = 128 if debug else hp.quniform('hiddens', 48,256, 1)
        args['clusters1'] = 32 if debug else hp.quniform('clusters1',1,256,1)
        args['clusters2'] = 6 if debug else hp.quniform('clusters2',0,128,1)
        args['clusters3'] = 0 if debug else hp.quniform('clusters3',0,64,1)
        args['visualizeClusters'] = False
        args['recalculate'] = False
        args['folds'] = 5        
        args['isAttentionLayer'] = 0 if debug else hp.choice("isAttentionLayer", [True,False])
        args['orthoColor'] = 1 if debug else hp.choice("orthoColor", [True,False])
        args['ortho'] = 1 if debug else hp.choice("ortho", [True,False])
        args['MinCut'] = 1 if debug else hp.choice("MinCut", [True,False])
        args['device'] = 'cuda:0'
        args['dropoutRate'] = 0.1 if debug else hp.uniform('dropoutRate', 0, 0.75)
        args['AttntnSparsenss'] = 1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0.1 if debug else hp.uniform('attntnThreshold', 0, 1)
        args['normalizeFeats'] = 0 if debug else hp.choice("normalizeFeats", [True,False])
        args['normalizeCells'] = 0 if debug else hp.choice("normalizeCells", [True,False])
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True   
        args['pearsonCoeffSUP'] = 1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = 1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])

    elif 'SyntheticV3' in path:                            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=128
        args['PCL_epochs']=3000
        args['PCL_patch_size']=8
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Labels_2classes'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 150
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 8# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.0001
        args['batch_size'] = 2 if debug=='Index' else hp.choice('batch_size', [2,6,12,16,20]) if debug=='Object' else 12
        args['lr'] = 1 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:2'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 1 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else False
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 2 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 64                
        args['clusters1'] = 1 if debug=='Index' else hp.choice('clusters1',[4,7,10]) if debug=='Object' else 7        
        args['clusters2'] = 1 if debug=='Index' else hp.choice('clusters2',[3,6,9]) if debug=='Object' else 6 
        args['clusters3'] = 1 if debug=='Index' else hp.choice('clusters3',[2,5,8]) if debug=='Object' else 5         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 0 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.05        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 1 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else .2     
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 1 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else False
        args['orthoColor_Lambda0'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001    
        args['min_Cell_entropy_Lambda1'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['min_Cell_entropy_Lambda2'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['MinCut'] = 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])        

        
    elif 'SyntheticV_H3' in path:        
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=128
        args['PCL_epochs']=3000
        args['PCL_patch_size']=10
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Labels'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 200
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 8# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.0001
        args['batch_size'] = 2 if debug=='Index' else hp.choice('batch_size', [2,6,12,16,20]) if debug=='Object' else 12
        args['lr'] = 1 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:3'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 1 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else False
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 2 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 64                
        args['clusters1'] = 1 if debug=='Index' else hp.choice('clusters1',[4,7,10,13,16]) if debug=='Object' else 7        
        args['clusters2'] = 1 if debug=='Index' else hp.choice('clusters2',[3,6,9,12,15]) if debug=='Object' else 6 
        args['clusters3'] = 1 if debug=='Index' else hp.choice('clusters3',[2,5,8,11,14]) if debug=='Object' else 5         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 0 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.05        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 1 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else .2     
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 1 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else False
        args['orthoColor_Lambda0'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001    
        args['min_Cell_entropy_Lambda1'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['min_Cell_entropy_Lambda2'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['MinCut'] = 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])        


    elif 'SyntheticV2' in path:
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=128
        args['PCL_epochs']=3000
        args['PCL_patch_size']=8
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Labels_Ex'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 500
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 2 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.01
        args['batch_size'] = 0 if debug=='Index' else hp.choice('batch_size', [6,12,16,20]) if debug=='Object' else 6
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:3'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 44                
        args['clusters1'] = 2 if debug=='Index' else hp.choice('clusters1',[4,7,10]) if debug=='Object' else 10        
        args['clusters2'] = 2 if debug=='Index' else hp.choice('clusters2',[3,6,9]) if debug=='Object' else 9 
        args['clusters3'] = 2 if debug=='Index' else hp.choice('clusters3',[2,5,8]) if debug=='Object' else 8         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 3 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.2        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else 0  
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 2 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 3                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 0 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1   
        args['min_Cell_entropy_Lambda1'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['min_Cell_entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])        

    elif 'SyntheticV_H2' in path:        
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=128
        args['PCL_epochs']=3000
        args['PCL_patch_size']=8
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Labels_2classes'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 500
        args['epochs'] =100# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 8# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.0001
        args['batch_size'] = 1 if debug=='Index' else hp.choice('batch_size', [2,6,12,16,20]) if debug=='Object' else 6
        args['lr'] = 1 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 2 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 64                
        args['clusters1'] = 3 if debug=='Index' else hp.choice('clusters1',[4,7,10,13,16]) if debug=='Object' else 13        
        args['clusters2'] = 2 if debug=='Index' else hp.choice('clusters2',[3,6,9,12,15]) if debug=='Object' else 9 
        args['clusters3'] = 4 if debug=='Index' else hp.choice('clusters3',[2,5,8,11,14]) if debug=='Object' else 14         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 1 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.1        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else 0     
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 3 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.0001                                
        args['orthoColor_Lambda1'] = 2 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001                              
        args['ortho'] = False#1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 0# 4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda1'] = 0# 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 0# 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01    
        args['min_Cell_entropy_Lambda1'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['min_Cell_entropy_Lambda2'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['MinCut'] = False# 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0# 0 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['MinCut_Lambda1'] = 0#1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0#0 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False]) 

    elif 'SyntheticV1' in path:        
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=128
        args['PCL_epochs']=3000
        args['PCL_patch_size']=12
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Labels_2classes'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 200
        args['epochs'] =20# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 7# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001
        args['batch_size'] = 2 if debug=='Index' else hp.choice('batch_size', [2,6,12,16,20]) if debug=='Object' else 12
        args['lr'] = 1 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:2'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False       
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True    
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [32,64,128]) if debug=='Object' else 64                
        args['clusters1'] = 4 if debug=='Index' else hp.choice('clusters1',[4,5,6,7,8]) if debug=='Object' else 8        
        args['clusters2'] = 4 if debug=='Index' else hp.choice('clusters2',[4,5,6,7,8]) if debug=='Object' else 10 
        args['clusters3'] = 2 if debug=='Index' else hp.choice('clusters3',[2,3,4,5,6]) if debug=='Object' else 7         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 0 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.05        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 1 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else .2     
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = False# 0 if debug else hp.choice("orthoColor", [True,False])
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001    
        args['min_Cell_entropy_Lambda1'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['min_Cell_entropy_Lambda2'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['MinCut'] = 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])
        
    elif 'SyntheticV_H1' in path:        
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=128
        args['PCL_epochs']=3000
        args['PCL_patch_size']=8
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Labels_2classes'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 200
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 7# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001
        args['batch_size'] = 2 if debug=='Index' else hp.choice('batch_size', [2,6,12,16,20]) if debug=='Object' else 12
        args['lr'] = 1 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:2'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False       
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True    
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [32,64,128]) if debug=='Object' else 64                
        args['clusters1'] = 4 if debug=='Index' else hp.choice('clusters1',[4,5,6,7,8]) if debug=='Object' else 8        
        args['clusters2'] = 4 if debug=='Index' else hp.choice('clusters2',[4,5,6,7,8]) if debug=='Object' else 10 
        args['clusters3'] = 2 if debug=='Index' else hp.choice('clusters3',[2,3,4,5,6]) if debug=='Object' else 7         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 0 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.05        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 1 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else .2     
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = False# 0 if debug else hp.choice("orthoColor", [True,False])
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001    
        args['min_Cell_entropy_Lambda1'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['min_Cell_entropy_Lambda2'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['MinCut'] = 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])
        

    elif 'SyntheticV4' in path:
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=180
        args['PCL_epochs']=1000
        args['PCL_patch_size']=10
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Labels_2classes'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 500
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 2 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.01
        args['batch_size'] = 0 if debug=='Index' else hp.choice('batch_size', [6,12,16,20]) if debug=='Object' else 6
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:3'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 44                
        args['clusters1'] = 2 if debug=='Index' else hp.choice('clusters1',[4,7,10]) if debug=='Object' else 10        
        args['clusters2'] = 2 if debug=='Index' else hp.choice('clusters2',[3,6,9]) if debug=='Object' else 9 
        args['clusters3'] = 2 if debug=='Index' else hp.choice('clusters3',[2,5,8]) if debug=='Object' else 8         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 3 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.2        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else 0  
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 2 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 3                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 0 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1   
        args['min_Cell_entropy_Lambda1'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['min_Cell_entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])        

    elif 'ZuriBaselRisk' in path:
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=100
        args['PCL_epochs']=500
        args['PCL_patch_size']=18
        args['PCL_alpha_L']=1.15 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 
        
        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Risk_3_classes', 'OSmonth','Alive'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 400
        args['epochs'] =30# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 7# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.0001
        args['batch_size'] = 3 if debug=='Index' else hp.choice('batch_size', [8,14,22,26,44,64,88,96]) if debug=='Object' else 26
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 11#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['normalizeFeats'] = 0 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else True        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False    
        args['Batch_Normalization'] =True#  0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True       
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 0 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05,0.1,0.15]) if debug=='Object' else 0   

        # Neural Network
        args['hiddens'] = 3 if debug=='Index' else hp.choice('hiddens', [32,44,64,96,128,256]) if debug=='Object' else 96                
        args['clusters1'] = 2 if debug=='Index' else hp.choice('clusters1',[10,13,16,19,22]) if debug=='Object' else 13        
        args['clusters2'] = 4 if debug=='Index' else hp.choice('clusters2',[9,12,15,18,21]) if debug=='Object' else 18 
        args['clusters3'] = 1 if debug=='Index' else hp.choice('clusters3',[8,11,14,17,20]) if debug=='Object' else 11       
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 1 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 4 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25,0.3,0.35,0.40]) if debug=='Object' else 0.25        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 2 if debug=='Index' else hp.choice('attntnThreshold', [0,0.1,0.2,0.3,0.4,0.5,0.6]) if debug=='Object' else .2       
        args['GraphConvolution'] = 'ResNet'#0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 0 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 1                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 1 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else False
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                       
        args['ortho_Lambda0'] = 4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                              
        args['ortho_Lambda1'] = 0 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda2'] = 1 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01                                
        args['min_Cell_entropy'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else False                                        
        args['min_Cell_entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else .0001   
        args['min_Cell_entropy_Lambda1'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['min_Cell_entropy_Lambda2'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1
        args['MinCut'] = 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False
        args['Lasso_Feat_Selection_Lambda0'] = 3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda1'] = 3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda1", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['SupervisedLearning_Lambda0'] = 3 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.001      
        args['SupervisedLearning_Lambda1'] = 5 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001         
        args['SupervisedLearning_Lambda2'] = 0#5 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001,0]) if debug=='Object' else 0         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])

    elif 'ZuriBaselType' in path:
        # SuperPatch
        args['GetSuperpatch'] = False
        args['Train_SuperPatchModel'] = False
        args['WholeSlide'] = False
        args['output_Dimensions'] = 256
        args['encoder'] = 'SIMCLR'
        args['train_batch_size']=120
        args['train_epochs']=500
        args['patch_size']=15
        args['ContextMultiplication']=1.15
        args['z-scoreNorm']=True
        
        # SuperPatch or SuperPixel        
        args['UseSuperpatch']=True

        # Optimization Parameters
        args['epochs'] =30# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 10# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)
        args['weight_decay'] = 2 if debug else hp.choice('weight_decay',[0.001,0.0005,0.00001])
        args['batch_size'] =16# 1 if debug else hp.choice('batch_size', [10])
        args['lr'] = 1 if debug else hp.choice('lr', [0.005,0.001,0.0005])
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 17#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 2
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = False
        args['recalculate'] = True
        args['folds'] = 10
        args['device'] = 'cuda'
        args['normalizeFeats'] = True# if debug else hp.choice("normalizeFeats", [True,False])
        args['normalizeCells'] = False# 1 if debug else hp.choice("normalizeCells", [True,False])
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True   
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 0 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001])

        # Neural Network
        args['hiddens'] = 0 if debug else hp.choice('hiddens', [128])
        args['clusters1'] = 1 if debug else hp.choice('clusters1',[12,24,48])
        args['clusters2'] = 1 if debug else hp.choice('clusters2',[18,32,54])
        args['clusters3'] = 1 if debug else hp.choice('clusters3',[16,22,42])  
        args['LSTM'] = False#1 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = True#0 if debug else hp.choice("GLORE", [True,False])
        args['Phenotypes'] = True             
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = True#0 if debug else hp.choice("ClusteringOrAttention", [True,False])
        args['1cell1cluster'] = True#0 if debug else hp.choice("1cell1cluster", [True,False])
        args['dropoutRate'] = 1 if debug else hp.choice('dropoutRate', [0.001,0.01,0.05])
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0#0 if debug else hp.choice('attntnThreshold', [0.01,0.3,0.45,0.6])        
        args['GraphConvolution'] = 'ResNet'#0 if debug else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) # 0:ResNet, 1:Inception
        args['n-hops'] = 3#1 if debug else hp.choice('n-hops', [2, 3]) # 0:2-hops, 1:3-hops
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = False#1 if debug else hp.choice("orthoColor", [True,False])
        args['ortho'] = False#1 if debug else hp.choice("ortho", [True,False])
        args['ortho_Lambda0'] = 1# if debug else hp.choice("ortho_Lambda0", [0.1,0.001,0.0001])
        args['ortho_Lambda1'] = 1# if debug else hp.choice("ortho_Lambda1", [0.1,0.001,0.0001])
        args['ortho_Lambda2'] = 1# if debug else hp.choice("ortho_Lambda2", [0.1,0.001,0.0001])
        args['min_Cell_entropy'] = True#1 if debug else hp.choice("ortho", [True,False])
        args['min_Cell_entropy_Lambda0'] = 1 if debug else hp.choice("min_Cell_entropy_Lambda0", [10,1,0.1,0.01])
        args['min_Cell_entropy_Lambda1'] = 1 if debug else hp.choice("min_Cell_entropy_Lambda1", [10,1,0.1,0.01])
        args['min_Cell_entropy_Lambda2'] = 0 if debug else hp.choice("min_Cell_entropy_Lambda2", [0.01,0.001,0.0001])
        args['MinCut'] = False#1 if debug else hp.choice("MinCut", [True,False])
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = True#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.5,0.01,0.001])
        args['Max_Pat_Entropy_Lambda1'] = 2 if debug else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.2,0.05,0.001])
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001])
        args['UnsupContrast'] = False# if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] = 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] = 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] = 3 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda0'] = 1 if debug else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda1'] = 1 if debug else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda2'] = 2 if debug else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda3'] = 0 if debug else hp.choice("SupervisedLearning_Lambda3", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])
        args['Lasso_Feat_Selection'] = False
        args['Lasso_Feat_Selection_Lambda0'] = 1 #= 0 if debug else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0.0001])

    elif 'ZuriBaselSurvival' in path:
        # SuperPatch
        args['GetSuperpatch'] = False
        args['Train_SuperPatchModel'] = False
        args['WholeSlide'] = False
        args['output_Dimensions'] = 256
        args['encoder'] = 'SIMCLR'
        args['train_batch_size']=120
        args['train_epochs']=500
        args['patch_size']=15
        args['ContextMultiplication']=1.15
        args['z-scoreNorm']=True
        
        # SuperPatch or SuperPixel        
        args['UseSuperpatch']=True

        # Optimization Parameters
        args['epochs'] =1# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 10# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)
        args['weight_decay'] = 2 if debug else hp.choice('weight_decay',[0.001,0.0005,0.00001])
        args['batch_size'] =14# 1 if debug else hp.choice('batch_size', [10])
        args['lr'] = 1 if debug else hp.choice('lr', [0.005,0.001,0.0005])
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 17#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 2
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = False
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['normalizeFeats'] = True# if debug else hp.choice("normalizeFeats", [True,False])
        args['normalizeCells'] = False# 1 if debug else hp.choice("normalizeCells", [True,False])
        args['Batch_Normalization'] = 1 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else False   
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 0 if debug=='Index' else hp.choice("dataAugmentationPerc", [0.0001,0])

        # Neural Network
        args['hiddens'] = 0 if debug else hp.choice('hiddens', [128])
        args['clusters1'] = 1 if debug else hp.choice('clusters1',[12,16,20])
        args['clusters2'] = 1 if debug else hp.choice('clusters2',[10,14,18])
        args['clusters3'] = 1 if debug else hp.choice('clusters3',[6,8,11])  
        args['LSTM'] = False#1 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = True#0 if debug else hp.choice("GLORE", [True,False])
        args['Phenotypes'] = True             
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = True#0 if debug else hp.choice("ClusteringOrAttention", [True,False])
        args['1cell1cluster'] = True#0 if debug else hp.choice("1cell1cluster", [True,False])
        args['dropoutRate'] = 1 if debug else hp.choice('dropoutRate', [0.001,0.01,0.05])
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0#0 if debug else hp.choice('attntnThreshold', [0.01,0.3,0.45,0.6])        
        args['GraphConvolution'] = 'ResNet'#0 if debug else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) # 0:ResNet, 1:Inception
        args['n-hops'] = 3#1 if debug else hp.choice('n-hops', [2, 3]) # 0:2-hops, 1:3-hops
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = False#1 if debug else hp.choice("orthoColor", [True,False])
        args['ortho'] = False#1 if debug else hp.choice("ortho", [True,False])
        args['ortho_Lambda0'] = 1# if debug else hp.choice("ortho_Lambda0", [0.1,0.001,0.0001])
        args['ortho_Lambda1'] = 1# if debug else hp.choice("ortho_Lambda1", [0.1,0.001,0.0001])
        args['ortho_Lambda2'] = 1# if debug else hp.choice("ortho_Lambda2", [0.1,0.001,0.0001])
        args['min_Cell_entropy'] = True#1 if debug else hp.choice("ortho", [True,False])
        args['min_Cell_entropy_Lambda0'] = 1 if debug else hp.choice("min_Cell_entropy_Lambda0", [10,1,0.1,0.01])
        args['min_Cell_entropy_Lambda1'] = 1 if debug else hp.choice("min_Cell_entropy_Lambda1", [10,1,0.1,0.01])
        args['min_Cell_entropy_Lambda2'] = 0 if debug else hp.choice("min_Cell_entropy_Lambda2", [0.01,0.001,0.0001])
        args['MinCut'] = False#1 if debug else hp.choice("MinCut", [True,False])
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = True#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.5,0.01,0.001])
        args['Max_Pat_Entropy_Lambda1'] = 2 if debug else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.2,0.05,0.001])
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001])
        args['UnsupContrast'] = False# if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] = 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] = 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] = 3 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda0'] = 1 if debug else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda1'] = 1 if debug else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda2'] = 2 if debug else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning_Lambda3'] = 0 if debug else hp.choice("SupervisedLearning_Lambda3", [1,0.1, 0.01,0.001,0.0001])
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])
        args['Lasso_Feat_Selection'] = False
        args['Lasso_Feat_Selection_Lambda0'] = 1 #= 0 if debug else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0.0001])

    
    elif 'ZuriBaselHRHER2' in path:
        # Optimization Parameters
        args['epochs'] = 20# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5 if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 2 if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)
        args['weight_decay'] = 0.01 if debug else hp.uniform('weight_decay', 0.01, 0.1)
        args['batch_size'] = 12 if debug else hp.quniform('batch_size', 12, 24, 1)
        args['lr'] = 0.001 if debug else hp.uniform('lr', 0.0001, 0.1)
        args['useOptimizer'] = 0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound
        
        # General
        args['context_size'] = 1 if debug else hp.choice("context_size", [5, 10, 30, 50])
        args['num_classes'] = 4
        args['visualizeClusters'] = False
        args['recalculate'] = False
        args['folds'] = 5        
        args['device'] = 'cuda:3'
        args['normalizeFeats'] = 0 if debug else hp.choice("normalizeFeats", [True,False])
        args['normalizeCells'] = 0 if debug else hp.choice("normalizeCells", [True,False])
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True   
        args['normalizePercentile'] = 0 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] =  0.01

        # Neural Network
        args['hiddens'] = 128# if debug else hp.quniform('hiddens', 48,256, 1)
        args['clusters1'] = 98 if debug else hp.quniform('clusters1',1,256,1)
        args['clusters2'] = 15 if debug else hp.quniform('clusters2',0,128,1)
        args['clusters3'] = 0 if debug else hp.quniform('clusters3',0,64,1)        
        args['isAttentionLayer'] = 1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['dropoutRate'] = 0.05 if debug else hp.uniform('dropoutRate', 0, 0.5)
        args['AttntnSparsenss'] = 0 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0.23 if debug else hp.uniform('attntnThreshold', 0, 1)        
        args['GraphConvolution'] = 1 if debug else hp.choice('GraphConvolution', ['ResNet', 'Inception']) # 0:ResNet, 1:Inception
        args['n-hops'] = 1 if debug else hp.choice('n-hops', [2, 3]) # 0:2-hops, 1:3-hops
        args['modeltype'] = 'SAGE'#1 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = 1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = 0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function        
        args['NearestNeighborClassification'] = 1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['KinNearestNeighbors'] = 0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
    
        # Losses
        args['pearsonCoeffSUP'] = 1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = 0 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 1 if debug else hp.choice("orthoColor", [True,False])
        args['ortho'] = 0 if debug else hp.choice("ortho", [True,False])
        args['MinCut'] = 0 if debug else hp.choice("MinCut", [True,False])

    elif 'LungCD8CD4FOXP3' in path:            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=100
        args['PCL_epochs']=500
        args['PCL_patch_size']=30
        args['PCL_alpha_L']=1.15 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 
        
        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Risk_2_classes_d'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 450
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.0001
        args['batch_size'] = 6 if debug=='Index' else hp.choice('batch_size', [8,14,22,26,44,64,88,96]) if debug=='Object' else 88
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 11#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False   
        args['Batch_Normalization'] =True#  0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True       
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 2 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05,0.1,0.15]) if debug=='Object' else 0.001  

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [16,32,44,64,96,128]) if debug=='Object' else 32                
        args['clusters1'] = 1 if debug=='Index' else hp.choice('clusters1',[4,7,10]) if debug=='Object' else 7        
        args['clusters2'] = 1 if debug=='Index' else hp.choice('clusters2',[3,6,9]) if debug=='Object' else 6 
        args['clusters3'] = 1 if debug=='Index' else hp.choice('clusters3',[2,5,8]) if debug=='Object' else 5        
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 1 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else False        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 7 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25,0.3,0.35,0.40]) if debug=='Object' else 0.4        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 4 if debug=='Index' else hp.choice('attntnThreshold', [0,0.1,0.2,0.3,0.4,0.5,0.6]) if debug=='Object' else .4
        args['GraphConvolution'] = 'ResNet'#0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 0 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 1
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 1 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else False
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = False#1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                       
        args['ortho_Lambda0'] = 0#4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                              
        args['ortho_Lambda1'] = 0# if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda2'] = 0# if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01                                
        args['min_Cell_entropy'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else False                                        
        args['min_Cell_entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1   
        args['min_Cell_entropy_Lambda1'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['min_Cell_entropy_Lambda2'] = 5 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0
        args['MinCut'] = False# 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0# 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 0 # 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0 #1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = False#1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 0 # 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 0 #  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 #  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = False#1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False
        args['Lasso_Feat_Selection_Lambda0'] = 0# 3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda1'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda1", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda2'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda2", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.1      
        args['SupervisedLearning_Lambda1'] = 4 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.0001         
        args['SupervisedLearning_Lambda2'] = 0#3 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.001         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])

    elif 'LungQKBRCGLUT' in path:            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=100
        args['PCL_epochs']=500
        args['PCL_patch_size']=30
        args['PCL_alpha_L']=1.15 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 
        
        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Risk_2_classes_d'] # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 500
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 4 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.0001
        args['batch_size'] = 6 if debug=='Index' else hp.choice('batch_size', [8,14,22,26,44,64,88]) if debug=='Object' else 88
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 11#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:0'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False   
        args['Batch_Normalization'] =True#  0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True       
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 0 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05,0.1,0.15]) if debug=='Object' else 0   

        # Neural Network
        args['hiddens'] = 0 if debug=='Index' else hp.choice('hiddens', [20,32,44,64,96,128]) if debug=='Object' else 32                
        args['clusters1'] = 3 if debug=='Index' else hp.choice('clusters1',[7,10,13,16]) if debug=='Object' else 16        
        args['clusters2'] = 2 if debug=='Index' else hp.choice('clusters2',[6,9,12,15]) if debug=='Object' else 12 
        args['clusters3'] = 3 if debug=='Index' else hp.choice('clusters3',[5,8,11,14]) if debug=='Object' else 14        
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 0 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else True 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 1 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else False        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 4 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.2,0.3,0.40]) if debug=='Object' else 0.4        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 4 if debug=='Index' else hp.choice('attntnThreshold', [0,0.1,0.2,0.3,0.4,0.5,0.6]) if debug=='Object' else .4
        args['GraphConvolution'] = 'ResNet'#0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 0 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 1
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 1 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else False
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = False#1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                       
        args['ortho_Lambda0'] = 0#4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                              
        args['ortho_Lambda1'] = 0# if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda2'] = 0# if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01                                
        args['min_Cell_entropy'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else False                                        
        args['min_Cell_entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1   
        args['min_Cell_entropy_Lambda1'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['min_Cell_entropy_Lambda2'] = 5 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0
        args['MinCut'] = False# 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0# 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 0 # 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0 #1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = False#1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False
        args['Lasso_Feat_Selection_Lambda0'] = 0# 3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda1'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda1", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda2'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda2", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.1      
        args['SupervisedLearning_Lambda1'] = 4 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.0001         
        args['SupervisedLearning_Lambda2'] = 0#3 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.001         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])

    elif 'Endometrial_LowGrade' in path:            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=180
        args['PCL_epochs']=1000
        args['PCL_patch_size']=20
        args['PCL_alpha_L']=1.1 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Relapse'] # Relapse, MLH1 methylated, Mismatch repair protein status, RASSF1A methylated # If it is a list of strings, multilabel classification is allowed                       

        # Optimization Parameters
        args['num_samples_architecture_search'] = 200
        args['epochs'] =15# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 1 if debug=='Index' else hp.choice('lr_decay_step_size', [0.25,0.5,0.75]) if debug=='Object' else 0.5  
        args['lr_decay_step_size'] = 1 if debug=='Index' else hp.choice('lr_decay_step_size', [4,8,12,14]) if debug=='Object' else 8        
        args['weight_decay'] = 2 if debug=='Index' else hp.choice('weight_decay',[0.001,0.0005,0.0001,0.00005,0.00001]) if debug=='Object' else 0.0001
        args['batch_size'] = 2 if debug=='Index' else hp.choice('batch_size', [8, 12, 16, 18]) if debug=='Object' else 16
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.01,0.005,0.001,0.0005,0.0001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 45#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 2
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = False
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['learnSupvsdClust'] = True
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False  
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True         
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc",[0.001,0.0001,0.00001,0]) if debug=='Object' else 0.0001        

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [64,96,128]) if debug=='Object' else 96                
        args['clusters1'] = 0 if debug=='Index' else hp.choice('clusters1',[10,14,20,28,36]) if debug=='Object' else 10        
        args['clusters2'] = 0 if debug=='Index' else hp.choice('clusters2',[8,12,18,26,34]) if debug=='Object' else 8 
        args['clusters3'] = 0 if debug=='Index' else hp.choice('clusters3',[4,8,16,24,32]) if debug=='Object' else 4         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 1 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else False        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 1 if debug=='Index' else hp.choice('dropoutRate', [0.001,0.01,0.05,0.1, 0.2]) if debug=='Object' else 0.01        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0#0 if debug else hp.choice('attntnThreshold', [0.01,0.3,0.45,0.6])        
        args['GraphConvolution'] = 'ResNet'#0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = False# 0 if debug else hp.choice("orthoColor", [True,False])
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 5 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [10,1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001    
        args['min_Cell_entropy_Lambda1'] = 3 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [10,1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01
        args['min_Cell_entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01
        args['MinCut'] = 1 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else False        
        args['MinCut_Lambda0'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 0 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.01        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])


    elif 'KIRC' in path:            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=180
        args['PCL_epochs']=1000
        args['PCL_patch_size']=50
        args['PCL_alpha_L']=1.1 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Type'] #

        # Optimization Parameters
        args['num_samples_architecture_search'] = 20
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 15# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 3 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['batch_size'] = 3 if debug=='Index' else hp.choice('batch_size', [8,14,22,26,44,64]) if debug=='Object' else 26
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 11#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False   
        args['Batch_Normalization'] =True#  0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True       
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 2 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05,0.1,0.15]) if debug=='Object' else 0.001   

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [20,32,44,64,96,128,180,256]) if debug=='Object' else 32                
        args['clusters1'] = 1 if debug=='Index' else hp.choice('clusters1',[7,10,13,16,19,22]) if debug=='Object' else 10        
        args['clusters2'] = 1 if debug=='Index' else hp.choice('clusters2',[6,9,12,15,18,21]) if debug=='Object' else 9 
        args['clusters3'] = 4 if debug=='Index' else hp.choice('clusters3',[5,8,11,14,17,20]) if debug=='Object' else 17        
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = False# 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 1 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else False        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 4 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.2,0.3,0.40]) if debug=='Object' else 0.4        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 4 if debug=='Index' else hp.choice('attntnThreshold', [0,0.1,0.2,0.3,0.4,0.5,0.6]) if debug=='Object' else .4
        args['GraphConvolution'] = 'ResNet'#0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = False#1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                       
        args['ortho_Lambda0'] = 0#4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                              
        args['ortho_Lambda1'] = 0# if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda2'] = 0# if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                   
        args['min_Cell_entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1   
        args['min_Cell_entropy_Lambda1'] = 3 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.001
        args['min_Cell_entropy_Lambda2'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda2'] = 2 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 0 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else True                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 3  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001        
        args['Max_Pat_Entropy_Lambda2'] = 1  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = False#1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False
        args['Lasso_Feat_Selection_Lambda0'] = 0# 3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda1'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda1", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda2'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda2", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.01      
        args['SupervisedLearning_Lambda1'] = 4 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.0001         
        args['SupervisedLearning_Lambda2'] = 0#3 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.001         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])
    elif 'GBM' in path:            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=180
        args['PCL_epochs']=1000
        args['PCL_patch_size']=30
        args['PCL_alpha_L']=1.1 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Type'] #

        # Optimization Parameters
        args['num_samples_architecture_search'] = 20
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 3 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['batch_size'] = 3 if debug=='Index' else hp.choice('batch_size', [8,14,22,26,44,64]) if debug=='Object' else 26
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 11#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False   
        args['Batch_Normalization'] =True#  0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True       
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 2 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05,0.1,0.15]) if debug=='Object' else 0.001   

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [20,32,44,64,96,128,180,256]) if debug=='Object' else 32                
        args['clusters1'] = 1 if debug=='Index' else hp.choice('clusters1',[7,10,13,16,19,22]) if debug=='Object' else 10        
        args['clusters2'] = 1 if debug=='Index' else hp.choice('clusters2',[6,9,12,15,18,21]) if debug=='Object' else 9 
        args['clusters3'] = 4 if debug=='Index' else hp.choice('clusters3',[5,8,11,14,17,20]) if debug=='Object' else 17        
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = False# 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 1 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else False        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 4 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.2,0.3,0.40]) if debug=='Object' else 0.4        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 4 if debug=='Index' else hp.choice('attntnThreshold', [0,0.1,0.2,0.3,0.4,0.5,0.6]) if debug=='Object' else .4
        args['GraphConvolution'] = 'ResNet'#0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = False#1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                       
        args['ortho_Lambda0'] = 0#4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                              
        args['ortho_Lambda1'] = 0# if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda2'] = 0# if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                   
        args['min_Cell_entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1   
        args['min_Cell_entropy_Lambda1'] = 3 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.001
        args['min_Cell_entropy_Lambda2'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda2'] = 2 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 0 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else True                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 3  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001        
        args['Max_Pat_Entropy_Lambda2'] = 1  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = False#1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False
        args['Lasso_Feat_Selection_Lambda0'] = 0# 3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda1'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda1", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda2'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda2", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.01      
        args['SupervisedLearning_Lambda1'] = 4 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.0001         
        args['SupervisedLearning_Lambda2'] = 0#3 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.001         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])


    elif 'Endometrial_POLE' in path:            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=80
        args['PCL_epochs']=1000
        args['PCL_patch_size']=15
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=50#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['POLE Mutation','Copy number variation','MSI Status','Tumour Type'] # Copy number variation, POLE Mutation, MSI Status, Tumour Type

        # Optimization Parameters
        args['num_samples_architecture_search'] = 500
        args['epochs'] =10# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 2 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.01
        args['batch_size'] = 0 if debug=='Index' else hp.choice('batch_size', [6,12,16,20]) if debug=='Object' else 6
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:3'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 44                
        args['clusters1'] = 2 if debug=='Index' else hp.choice('clusters1',[4,7,10]) if debug=='Object' else 10        
        args['clusters2'] = 2 if debug=='Index' else hp.choice('clusters2',[3,6,9]) if debug=='Object' else 9 
        args['clusters3'] = 1 if debug=='Index' else hp.choice('clusters3',[2,7,8]) if debug=='Object' else 7          
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 3 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.2        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else 0  
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 2 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 3                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 0 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1   
        args['min_Cell_entropy_Lambda1'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['min_Cell_entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda1'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda2'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda3'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda3", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])   


    elif 'Parkinson' in path:            
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=80
        args['PCL_epochs']=1000
        args['PCL_patch_size']=15
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['Status'] # Copy number variation, POLE Mutation, MSI Status, Tumour Type
        # ResNet
        args['model_depth'] = 101

        # Optimization Parameters
        args['num_samples_architecture_search'] = 500
        args['epochs'] =50# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 2 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.01
        args['batch_size'] = 1 if debug=='Index' else hp.choice('batch_size', [6,10,16,20]) if debug=='Object' else 10
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:0'
        args['normalizeFeats'] = 0 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else True        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 44                
        args['clusters1'] = 2 if debug=='Index' else hp.choice('clusters1',[4,7,10]) if debug=='Object' else 10        
        args['clusters2'] = 2 if debug=='Index' else hp.choice('clusters2',[3,6,9]) if debug=='Object' else 9 
        args['clusters3'] = 2 if debug=='Index' else hp.choice('clusters3',[2,5,8]) if debug=='Object' else 8         
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 3 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.2        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else 0  
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 2 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 3                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 0 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1   
        args['min_Cell_entropy_Lambda1'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['min_Cell_entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda1'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda2'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda3'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda3", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])   


    elif 'KIRC' in path:        
        # SuperPatch
        args['GetSuperpatch'] = False
        args['Train_SuperPatchModel'] = False
        args['WholeSlide'] = False
        args['output_Dimensions'] = 256
        args['encoder'] = 'SIMCLR'
        args['train_batch_size']=128
        args['train_epochs']=6000
        args['patch_size']=12
        args['ContextMultiplication']=1.25

        # SuperPatch or SuperPixel        
        args['UseSuperpatch']=False
        
        # Optimization Parameters
        args['epochs'] = 70# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 10# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)
        args['weight_decay'] = 0.001 if debug else hp.uniform('weight_decay', 0.01, 0.1)
        args['batch_size'] = 10 if debug else hp.quniform('batch_size', 8, 35, 1)
        args['lr'] = 0.001 if debug else hp.uniform('lr', 0.0001, 0.1)
        args['useOptimizer'] = 0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 1 if debug else hp.choice("context_size", [10, 15, 50])
        args['num_classes'] = 3
        args['visualizeClusters'] = False
        args['recalculate'] = False
        args['folds'] = 10        
        args['device'] = 'cuda:0'
        args['normalizeFeats'] = 0 if debug else hp.choice("normalizeFeats", [True,False])
        args['normalizeCells'] = 0 if debug else hp.choice("normalizeCells", [True,False])
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True   
        args['normalizePercentile'] = 1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] =  0.01

        # Neural Network
        args['hiddens'] = 128# if debug else hp.quniform('hiddens', 48,256, 1)
        args['clusters1'] = 32 if debug else hp.quniform('clusters1',1,256,1)
        args['clusters2'] = 8 if debug else hp.quniform('clusters2',0,128,1)
        args['clusters3'] = 0 if debug else hp.quniform('clusters3',0,64,1)        
        args['isAttentionLayer'] = 1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['dropoutRate'] = 0.15 if debug else hp.uniform('dropoutRate', 0, 0.5)
        args['AttntnSparsenss'] = 1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 1 if debug else hp.uniform('attntnThreshold', 0, 1)        
        args['GraphConvolution'] = 0 if debug else hp.choice('GraphConvolution', ['ResNet', 'Inception']) # 0:ResNet, 1:Inception
        args['n-hops'] = 1 if debug else hp.choice('n-hops', [2, 3]) # 0:2-hops, 1:3-hops
        args['modeltype'] = 0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = 1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = 0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = 1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['KinNearestNeighbors'] = 0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = 1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = 1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 1 if debug else hp.choice("orthoColor", [True,False])
        args['ortho'] = 0 if debug else hp.choice("ortho", [True,False])
        args['MinCut'] = 0 if debug else hp.choice("MinCut", [True,False])

    elif 'MouseBreast' in path:                
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=80
        args['PCL_epochs']=1000
        args['PCL_patch_size']=80
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # SuperPatch or SuperPixel        
        args['UseSuperpatch']=False
        args['experiment_Label']=['Radiation']

       # Optimization Parameters
        args['num_samples_architecture_search'] = 20
        args['epochs'] =40# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 3 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['batch_size'] = 3 if debug=='Index' else hp.choice('batch_size', [8,14,22,26,44,64]) if debug=='Object' else 26
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 11#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 10
        args['device'] = 'cuda:1'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False   
        args['Batch_Normalization'] =True#  0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True       
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 2 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.05,0.1,0.15]) if debug=='Object' else 0.001   

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [20,32,44,64,96,128,180,256]) if debug=='Object' else 32                
        args['clusters1'] = 1 if debug=='Index' else hp.choice('clusters1',[7,10,13,16,19,22]) if debug=='Object' else 10        
        args['clusters2'] = 1 if debug=='Index' else hp.choice('clusters2',[6,9,12,15,18,21]) if debug=='Object' else 9 
        args['clusters3'] = 4 if debug=='Index' else hp.choice('clusters3',[5,8,11,14,17,20]) if debug=='Object' else 17        
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = False# 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 1 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else False        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 4 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.2,0.3,0.40]) if debug=='Object' else 0.4        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 4 if debug=='Index' else hp.choice('attntnThreshold', [0,0.1,0.2,0.3,0.4,0.5,0.6]) if debug=='Object' else .4
        args['GraphConvolution'] = 'ResNet'#0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 1 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 2
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = False#1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                       
        args['ortho_Lambda0'] = 0#4 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                              
        args['ortho_Lambda1'] = 0# if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda2'] = 0# if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                   
        args['min_Cell_entropy_Lambda0'] = 1 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1   
        args['min_Cell_entropy_Lambda1'] = 3 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.001
        args['min_Cell_entropy_Lambda2'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda2'] = 2 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 0 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else True                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 3  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001        
        args['Max_Pat_Entropy_Lambda2'] = 1  if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = False#1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False
        args['Lasso_Feat_Selection_Lambda0'] = 0# 3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda1'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda1", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['Lasso_Feat_Selection_Lambda2'] = 0 #3 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda2", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.001         
        args['SupervisedLearning_Lambda0'] = 1 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.01      
        args['SupervisedLearning_Lambda1'] = 4 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.0001         
        args['SupervisedLearning_Lambda2'] = 0#3 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 0.001         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])


    else:
        # Patch contrastive learning parameters
        args['PCL_embedding_dimensions'] = 256
        args['PCL_batch_size']=80
        args['PCL_epochs']=1000
        args['PCL_patch_size']=15
        args['PCL_alpha_L']=1.2 # The value of alpha_L in the manuscript
        args['PCL_ZscoreNormalization']=True        
        args['PCL_width_CNN']=2 # [1, 2, 4]           
        args['PCL_depth_CNN']=101#4 # [1, 2, 4] 

        # Label you want to infer with respect the images.
        args['experiment_Label'] = ['POLE Mutation','Copy number variation','MSI Status','Tumour Type'] # Copy number variation, POLE Mutation, MSI Status, Tumour Type

        # Optimization Parameters
        args['num_samples_architecture_search'] = 2
        args['epochs'] =10# if debug else hp.quniform('epochs', 5, 25, 1)
        args['epoch'] = 0
        args['lr_decay_factor'] = 0.5# if debug else hp.uniform('lr_decay_factor', 0, 0.75)
        args['lr_decay_step_size'] = 12# if debug else hp.quniform('lr_decay_step_size', 2, 20, 1)        
        args['weight_decay'] = 2 if debug=='Index' else hp.choice('weight_decay',[1,0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.01
        args['batch_size'] = 0 if debug=='Index' else hp.choice('batch_size', [6,12,16,20]) if debug=='Object' else 6
        args['lr'] = 2 if debug=='Index' else hp.choice('lr', [0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.001
        args['useOptimizer'] = 'ADAM' #0 if debug else hp.choice("useOptimizer", ['ADAM', 'ADAMW', 'ADABound']) # 0:ADAM, 1:ADAMW, 2:ADABound

        # General
        args['context_size'] = 15#0 if debug else hp.choice("context_size", [15])
        args['num_classes'] = 3
        args['MultiClass_Classification']=1
        args['showHowNetworkIsTraining'] = False # Creates a GIF of the learning clusters!
        args['visualizeClusters'] = True
        args['learnSupvsdClust'] = True
        args['recalculate'] = False
        args['folds'] = 6
        args['device'] = 'cuda:3'
        args['normalizeFeats'] = 1 if debug=='Index' else hp.choice("normalizeFeats", [True,False]) if debug=='Object' else False        
        args['normalizeCells'] = 1 if debug=='Index' else hp.choice("normalizeCells", [True,False]) if debug=='Object' else False        
        args['Batch_Normalization'] = 0 if debug=='Index' else hp.choice("Batch_Normalization", [True,False]) if debug=='Object' else True
        args['normalizePercentile'] = False#1 if debug else hp.choice("normalizePercentile", [True,False])
        args['dataAugmentationPerc'] = 1 if debug=='Index' else hp.choice("dataAugmentationPerc", [0,0.0001,0.001,0.01,0.1]) if debug=='Object' else 0.0001    

        # Neural Network
        args['hiddens'] = 1 if debug=='Index' else hp.choice('hiddens', [32,44,64,86,128]) if debug=='Object' else 44                
        args['clusters1'] = 2 if debug=='Index' else hp.choice('clusters1',[4,7,10]) if debug=='Object' else 10        
        args['clusters2'] = 2 if debug=='Index' else hp.choice('clusters2',[3,6,9]) if debug=='Object' else 9 
        args['clusters3'] = 1 if debug=='Index' else hp.choice('clusters3',[2,7,8]) if debug=='Object' else 7          
        args['LSTM'] = False#0 if debug else hp.choice("LSTM", [True,False])
        args['GLORE'] = 1 if debug=='Index' else hp.choice('GLORE',[True,False]) if debug=='Object' else False 
        args['Phenotypes'] = True  
        args['DeepSimple'] = False
        args['isAttentionLayer'] = False#1 if debug else hp.choice("isAttentionLayer", [True,False])
        args['ClusteringOrAttention'] = 0 if debug=='Index' else hp.choice("ClusteringOrAttention", [True,False]) if debug=='Object' else True        
        args['1cell1cluster'] = 1 if debug=='Index' else hp.choice("1cell1cluster", [True,False]) if debug=='Object' else False                
        args['dropoutRate'] = 3 if debug=='Index' else hp.choice('dropoutRate', [0.05, 0.1, 0.15, 0.2,0.25]) if debug=='Object' else 0.2        
        args['AttntnSparsenss'] = False#1 if debug else hp.choice("AttntnSparsenss", [True,False])
        args['attntnThreshold'] = 0 if debug=='Index' else hp.choice('attntnThreshold', [0,.2,.4,.6,.8]) if debug=='Object' else 0  
        args['GraphConvolution'] = 0 if debug=='Index' else hp.choice('GraphConvolution', ['ResNet', 'IncepNet', 'JKNet']) if debug=='Object' else 'ResNet'                
        args['n-hops'] = 2 if debug=='Index' else hp.choice('n-hops', [1, 2, 3]) if debug=='Object' else 3                        
        args['modeltype'] = 'SAGE'#0 if debug else hp.choice('modeltype', ['SAGE', 'SGC']) # 0:SAGE, 1:SGC
        args['ObjectiveCluster'] = True#1 if debug else hp.choice('ObjectiveCluster', [True, False]) # Whether to learn a X and S embedding or just the clustering
        args['ReadoutFunction'] = False#0 if debug else hp.choice('ReadoutFunction', ['MAX', 'SUM', 'DeepSets']) # Choose the readout function    
        args['NearestNeighborClassification'] = False#1 if debug else hp.choice('NearestNeighborClassification', [True, False]) # Use the nearest Neighbor strategy
        args['NearestNeighborClassification_Lambda0'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda0", [0.1, 0.01 ,0.001, 0.0001])
        args['NearestNeighborClassification_Lambda1'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda1", [0.1, 0.01, 0.001, 0.0001])
        args['NearestNeighborClassification_Lambda2'] = 1#0 if debug else hp.choice("NearestNeighborClassification_Lambda2", [0.1, 0.01, 0.001, 0.0001])
        args['KinNearestNeighbors'] = 5#0 if debug else hp.choice('KinNearestNeighbors', [5, 10]) # Choose number of K in nearest neighbor strategy
        # Losses
        args['pearsonCoeffSUP'] = False#1 if debug else hp.choice("pearsonCoeffSUP", [True,False])
        args['pearsonCoeffUNSUP'] = False#1 if debug else hp.choice("pearsonCoeffUNSUP", [True,False])
        args['orthoColor'] = 0 if debug=='Index' else hp.choice("orthoColor", [True,False]) if debug=='Object' else True
        args['orthoColor_Lambda0'] = 0 if debug=='Index' else hp.choice("orthoColor_Lambda0", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.1                                
        args['orthoColor_Lambda1'] = 4 if debug=='Index' else hp.choice("orthoColor_Lambda1", [0.1,0.01,0.001,0.0001,0.00001]) if debug=='Object' else 0.00001                              
        args['ortho'] = 1 if debug=='Index' else hp.choice("ortho", [True,False]) if debug=='Object' else False                        
        args['ortho_Lambda0'] = 0 if debug=='Index' else hp.choice("ortho_Lambda0", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1                                
        args['ortho_Lambda1'] = 4 if debug=='Index' else hp.choice("ortho_Lambda1", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['ortho_Lambda2'] = 4 if debug=='Index' else hp.choice("ortho_Lambda2", [0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0                                
        args['min_Cell_entropy'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy", [True,False]) if debug=='Object' else True                                        
        args['min_Cell_entropy_Lambda0'] = 0 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 1   
        args['min_Cell_entropy_Lambda1'] = 4 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.0001
        args['min_Cell_entropy_Lambda2'] = 2 if debug=='Index' else hp.choice("min_Cell_entropy_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.01
        args['MinCut'] = 0 if debug=='Index' else hp.choice("MinCut", [True,False]) if debug=='Object' else True        
        args['MinCut_Lambda0'] = 5 if debug=='Index' else hp.choice("MinCut_Lambda0", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0        
        args['MinCut_Lambda1'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda1", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['MinCut_Lambda2'] = 1 if debug=='Index' else hp.choice("MinCut_Lambda2", [1,0.1,0.01,0.001,0.0001,0]) if debug=='Object' else 0.1        
        args['F-test'] = False#1 if debug else hp.choice("F-test", [True,False])
        args['Max_Pat_Entropy'] = 1 if debug=='Index' else hp.choice('Max_Pat_Entropy', [True, False]) if debug=='Object' else False                
        args['Max_Pat_Entropy_Lambda0'] = 4 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda0", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.0001        
        args['Max_Pat_Entropy_Lambda1'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda1", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['Max_Pat_Entropy_Lambda2'] = 1 if debug=='Index' else hp.choice("Max_Pat_Entropy_Lambda2", [1,0.1,0.01,0.001,0.0001]) if debug=='Object' else 0.1        
        args['UnsupContrast'] = False#1 if debug else hp.choice("UnsupContrast", [True,False])
        args['UnsupContrast_Lambda0'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda0", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda1'] =0# 1 if debug else hp.choice("UnsupContrast_Lambda1", [1,0.1,0.01,0.001,0.0001])
        args['UnsupContrast_Lambda2'] =0# 2 if debug else hp.choice("UnsupContrast_Lambda2", [1,0.1,0.01,0.001,0.0001])        
        args['Lasso_Feat_Selection'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection", [True,False]) if debug=='Object' else False         
        args['Lasso_Feat_Selection_Lambda0'] = 1 if debug=='Index' else hp.choice("Lasso_Feat_Selection_Lambda0", [1,0.1, 0.01,0.001,0]) if debug=='Object' else 0.1         
        args['SupervisedLearning_Lambda0'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda0", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda1'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda1", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda2'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda2", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning_Lambda3'] = 0 if debug=='Index' else hp.choice("SupervisedLearning_Lambda3", [1,0.1, 0.01,0.001,0.0001]) if debug=='Object' else 1         
        args['SupervisedLearning'] = True# if debug else hp.choice("SupervisedLearning", [True,False])   

    return args
