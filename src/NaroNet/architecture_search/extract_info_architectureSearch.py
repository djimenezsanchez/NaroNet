import os 
import pandas as pd
import json
import numpy as np
import copy
import seaborn as sns
from NaroNet.BioInsights.add_annotation_stat import add_stat_annotation
import matplotlib.pyplot as plt
import os 
import pandas as pd
import json
import numpy as np
import copy
import seaborn as snss
import matplotlib.pyplot as plt
from itertools import combinations

def perf_by_pairs(save_dir, topk, file_name, frame_info):
    """
    Heatmap showing the performance by pairs.
    """
    SelectedCombinations = ['GLORE=True','ClusteringOrAttention=True','1cell1cluster=True','ortho=True','min_Cell_entropy=True','MinCut=True','Max_Pat_Entropy=True','Lasso_Feat_Selection=True']
    frame_Interp_Cop = []
    for fpa_1st in frame_info:
        for fpa_2nd in frame_info:  
            # In case those two consist of a combination of interesting modules.  
            if (fpa_1st in SelectedCombinations) and (fpa_2nd in SelectedCombinations):
                PosPairs = [frame_info[fpa_1st][idx] for idx in range(len(frame_info[fpa_1st])) if (frame_info[fpa_1st][idx]!=-1000 and frame_info[fpa_2nd][idx]!=-1000)]
                PosPairs.sort()
                if 'Interp' in file_name:
                    frame_Interp_Cop.append(PosPairs[-topk:])
                else:
                    frame_Interp_Cop.append(PosPairs[:topk])
    # Generate Heatmap
    heatmap = np.zeros((len(SelectedCombinations),len(SelectedCombinations)))
    n_iter = 0
    for n_1st, comb_1st in enumerate(SelectedCombinations):
        for n_2nd, comb_2nd in enumerate(SelectedCombinations):
            heatmap[n_1st,n_2nd] = sum(frame_Interp_Cop[n_iter])/topk
            n_iter+=1
    plt.close()
    ax = sns.heatmap(heatmap, annot=True)
    ax.set_xticklabels(['Global reasoning unit','Softmax activation function','Max-pooling','Orthogonal Loss','Patch entropy loss', 'Mincut loss','Patient entropy loss','Lasso loss (last layers)'],rotation=90)
    ax.set_yticklabels(['Global reasoning unit','Softmax activation function','Max-pooling','Orthogonal Loss','Patch entropy loss', 'Mincut loss','Patient entropy loss','Lasso loss (last layers)'],rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir+file_name+'_heatmapPairs.png',dpi=600)

def showBoxplot(save_dir,pandasDF,parameters,ylabel,file_name):
    '''
        Show boxplot
    '''
    plt.close()
    ax = sns.boxplot(data=pandasDF, order=parameters)
    ax.tick_params(labelsize=10)
    ax.set_xticklabels([p.split('=')[1]for p in parameters], fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(parameters[0].split('=')[0], fontsize=18)
    test_results = add_stat_annotation(ax, data=pandasDF, x=None, y=None, order=parameters,
                                    box_pairs=list(combinations(parameters,2)),
                                    test='Mann-Whitney', text_format='star',
                                    loc='inside', verbose=2)
    plt.savefig(save_dir+file_name+'_'+parameters[0].split('=')[0]+'_stats.png',dpi=600)

def obtain_configuration_names(load_dir):
    '''
        Obtain configuration and save it in frame_PredAcc
    '''
    
    # Initialize configruration 
    frame_PredAcc = {}
    init_list = list(-1000*np.ones(len(os.listdir(load_dir))))

    # Traverse load_dir    
    for n_run, dname in enumerate(os.listdir(load_dir)):
        
        # Load only json result files.
        if ('.json' in dname) or ('.xlsx' in dname) or ('.png' in dname) or ('.tmp' in dname) or ('.pkl' in dname):
            continue

        # Open Json Results file    
        with open(load_dir+dname+'/result.json', 'r') as json_file:        
            
            # Read each line from the results file.
            lines = json_file.readlines()
            if len(lines)>0: # Check if runs were finished
                data = json.loads(lines[-1])                     
                
                # Obtain names of the configuration (e.g. "1cell1cluster=True")
                for d in data['config']:
                    frame_PredAcc[d+'='+str(data['config'][d])] = copy.deepcopy(init_list)
    frame_Interp = copy.deepcopy(frame_PredAcc)

    return frame_PredAcc, frame_Interp

def obtain_configuration_performance(load_dir,frame_PredAcc,frame_Interp,metric):
    '''
        Assign performance to configurations
    '''

    # Traverse load_dir    
    for n_run, dname in enumerate(os.listdir(load_dir)):
        
        # Load only json result files.
        if ('.json' in dname) or ('.xlsx' in dname) or ('.png' in dname) or ('.tmp' in dname) or ('.pkl' in dname):
            continue
        
        # Open Json Results file    
        with open(load_dir+dname+'/result.json', 'r') as json_file:        
            lines = json_file.readlines()
            
            # Check if runs were finished
            if len(lines)>0: 
                data = json.loads(lines[-1])                                 
                
                # For each possible configuration check if this run has been chosen      
                for fr in frame_PredAcc:
                    
                    # has this configuration been chosen by the run?
                    if fr.split('=')[1] == str(data['config'][fr.split('=')[0]]):
                        # Assign the val loss
                        frame_PredAcc[fr][n_run]=data[metric]
                        if 'Synthetic' in load_dir:
                            frame_Interp[fr][n_run]=data['interpretability']*100                        
    return frame_PredAcc, frame_Interp

def save_architecture_search_stats(save_dir,load_dir,topk):
    '''
        Calculate statistics showing which NaroNet architectures perform best
    '''
    
    # Load configurations
    frame_PredAcc, frame_Interp = obtain_configuration_names(load_dir)
    frame_PredAcc, frame_Interp = obtain_configuration_performance(load_dir,frame_PredAcc,frame_Interp,'test_Cross_entropy')

    # Save run configurations to excel (just in case)
    pd.DataFrame.from_dict(frame_PredAcc).to_excel(load_dir+'/predAcc.xlsx')
    if 'Synthetic' in load_dir:
        pd.DataFrame.from_dict(frame_Interp).to_excel(load_dir+'/interp.xlsx')

    # Obtain performance by pairs of parameters.
    # perf_by_pairs(save_dir, topk, 'Accuracy', frame_PredAcc)
    # if 'Synthetic' in load_dir:
    #     perf_by_pairs(save_dir,topk, 'Interp', frame_Interp)

    # Take best performing architectures per parameters.
    conf_param_acc, topk_acc = topk_performing_architectures(frame_PredAcc, mode='min',topk=topk)    
    if 'Synthetic' in load_dir:
        conf_param_int, topk_int = topk_performing_architectures(frame_Interp, mode='min',topk=topk)
            
    # Display boxplots showing the performance 
    for p in conf_param_acc:
        showBoxplot(save_dir,pd.DataFrame.from_dict(topk_acc),p,ylabel='Cross-validation test loss',file_name='Acc')
    if 'Synthetic' in load_dir:
        for p in conf_param_int:
            showBoxplot(save_dir,pd.DataFrame.from_dict(topk_int),p,ylabel='Interpretability Accuracy (%)',file_name='Interp')

def topk_performing_architectures(frame, mode,topk):
    '''
        Obtain Best performing values for each test type with respect the interpretability
    '''
    
    # Initialize
    frame_topk = {}

    # Take the topk performing values.
    for fpa in frame:    
        listt = [i for i in frame[fpa] if i!=-1000]
        listt.sort()
        if len(listt)>=topk:        
            if mode=='min':
                frame_topk[fpa] = [i if i!=-1 else 0 for i in listt[:topk]]
            elif mode=='max':
                frame_topk[fpa] = [i if i!=-1 else 0 for i in listt[-topk:]]

    # List configurations that were chosen.
    parameters_all = []
    for fpa_init in frame_topk:
        fpa_init_type = fpa_init.split('=')[0]
        parameters = []
        for fpa in frame_topk:
            if fpa.split('=')[0]==fpa_init_type:
                parameters.append(fpa)
        if len(parameters)>1:
            parameters_all.append(parameters)
    
    return parameters_all, frame_topk


def extract_best_result(load_dir,metric,best_params):
    '''
        Method  that extracts the best result out of all the executions 
    '''
    n_runs = 0
    saved_metric = 0 if metric == 'acc_test' else 10

    # Assign runs performance to each parameter.
    for n_run, dname in enumerate(os.listdir(load_dir)):

        if ('.json' in dname) or ('.xlsx' in dname) or ('.png' in dname) or ('.pkl' in dname) or ('.tmp' in dname):
            continue

        # Open Json Results file    
        with open(load_dir+'/'+dname+'/result.json', 'r') as json_file:            
            
            lines = json_file.readlines()
            if len(lines)>0: # Check if runs were finished
                n_runs +=1
                data = json.loads(lines[-1])         
                # print(str(data[metric]))
                if saved_metric>data[metric]:                
                    print(metric+": "+str(data[metric])+'\ntrain_Cross_entropy:'+str(data['train_Cross_entropy'])+'\nacc_test:'+str(data['acc_test'])+'\ntrain_acc:'+str(data['train_acc']))
                    saved_metric = data[metric]
                    config = data['config']
    
    if 'config' in locals():
        return config, n_runs
    else:
        return best_params, n_runs


if __name__ == "__main__":
    load_dir = '/home/djsanchez/ray_results/ExplainableML/'
    save_architecture_search_stats(load_dir,load_dir,10)