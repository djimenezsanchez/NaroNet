import os
from DatasetParameters import CustomParameters
import NaroNet
import torch
import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
import time
import matplotlib.pyplot as plt
import traceback

from CNNRepLearn.featurize_wsi import featurize_image
from tensorboard import program
from CNNRepLearn.preprocess_images import preprocess_images
from CNNRepLearn.patch_contrastive_learning import patch_contrastive_learning
# from extractSuperpixel import extractSuperpixel
from hyperopt import hp, space_eval
import tensorflow as tf
from extract_info_architectureSearch import extract_best_result
from extract_info_architectureSearch import save_architecture_search_stats

class NaroNet_Search(Trainable):

    def _setup(self, parameters):  
        self.device = torch.device(parameters["device"] if torch.cuda.is_available() else "cpu")
        # Initialize 
        self.N = NaroNet.NaroNet(parameters, self.device)
        self.N.epoch = 0
              
    def _train(self):
        try:
            result = self.N.epoch_validation()
        except Exception as exc:
            del self.N                                   
            result = {"val_acc":0.1,"epoch": 1000,"maximize_acc_interpretability":0,"acc_test":0,"interpretability":0,"train_Cell_ent":0,"train_Cross_entropy":0,"train_Pat_ent":0,"train_UnsupContrast_acc":0,"train_acc":0,"train_loss":0,"val_Cell_ent":0,"val_Pat_ent":0,"val_UnsupContrast_acc":0,"val_loss":0,'val_Cross_entropy':10,'test_Cross_entropy':10}             
            pass            
        return result

    def _save(self, checkpoint_dir):        
        return

def run_NaroNet(path,parameters):
    '''
    Code to run NaroNet using the enriched graph.  
    '''

    # Set the device to run the Neural Network.
    device =  torch.device(parameters["device"] if torch.cuda.is_available() else "cpu")

    # Load the model.
    N = NaroNet.NaroNet(parameters, device)
    N.epoch = 0

    # Execute k-fold cross-validation
    n_validation_samples = parameters['batch_size']
    n_validation_samples = 1
    N.cross_validation(n_validation_samples)   

def get_BioInsights(path, parameters):
    '''
    Code to calculate and obtain all the statistics from the experiment.
    '''
    # Load the model.
    N = NaroNet.NaroNet(parameters, 'cpu')
    N.epoch = 0    
    N.dataset.args = parameters

    # Visualize results
    N.visualize_results()

def architecture_search(path,best_parameters,possible_parameters):
    
    # Metric to optimize
    metric ='maximize_acc_interpretability' if 'Synthetic' in path else 'test_Cross_entropy' #  "loss_test"
    #metric = 'test_Cross_entropy'
    num_gpus = 1 if 'Synthetic' in path else 4
    best_parameters['device'] = best_parameters['device'] if 'Synthetic' in path else 'cuda'
    possible_parameters['device'] = possible_parameters['device'] if 'Synthetic' in path else 'cuda'

    architecture_search_path = path+'Architecture_Search/'
    architecture_search_path_save = architecture_search_path+'Results/'
    if (not os.path.exists(architecture_search_path)) or (not os.path.exists(architecture_search_path+'NaroNet_Search/')) or (not os.path.exists(architecture_search_path_save)):        
        os.mkdir(architecture_search_path)
        os.mkdir(architecture_search_path+'NaroNet_Search/')
        os.mkdir(architecture_search_path_save)
    else:
        best_result, n_runs = extract_best_result(architecture_search_path+'NaroNet_Search/',metric,best_parameters)
        if (n_runs>10 and len(os.listdir(architecture_search_path_save))==0):
            save_architecture_search_stats(architecture_search_path_save,architecture_search_path+'NaroNet_Search/',5)
        if n_runs>best_parameters['num_samples_architecture_search']*0.9:
            return best_result

    # Restart Ray defensively in case the ray connection is lost. 
    ray.shutdown()  
    ray.init(local_mode=False, num_gpus=num_gpus,_redis_password="!YHLQMDLG!_#potter#_#717^_()hisltxo")#, gpu_ids=[1, 2]) # Local address
    
    # Set Scheduler
    scheduler=ASHAScheduler(time_attr="epoch",max_t=best_parameters['epochs']-2, metric=metric, mode="min")        
    
    # Set algorithm
    search_algo = HyperOptSearch(possible_parameters, metric=metric, mode="min",n_initial_points=int(best_parameters['num_samples_architecture_search']*0.75), points_to_evaluate=[best_parameters])    
    search_algo = ConcurrencyLimiter(search_algo, max_concurrent=num_gpus if num_gpus>0 else 1)

    # Running parameters
    runIt = {"num_samples": best_parameters['num_samples_architecture_search'], 
    "resources_per_trial":{"gpu": 1 if num_gpus>0 else 0, "cpu":1}, "checkpoint_freq":0, "local_dir":architecture_search_path} 
    sync_config = tune.SyncConfig()
    
    # Obtain the best hyperparameters
    analysis = tune.run(NaroNet_Search,scheduler=scheduler,search_alg=search_algo,sync_config=sync_config,**runIt)

    # Obtain best result parameters
    best_result, n_runs = extract_best_result(architecture_search_path+'NaroNet_Search/',metric,best_parameters)
    return best_result

def run(path):
    # Select Experiment
    parameters = CustomParameters(path, 'Value')
    possible_parameters = CustomParameters(path, 'Object')
    best_parameters = CustomParameters(path, 'Index')

    print('Executing experiment: ' + path)

    # Preprocess Images
    # preprocess_images(path,parameters['PCL_ZscoreNormalization'],parameters['PCL_patch_size'])

    # Patch Contrastive Learning
    # patch_contrastive_learning(path,parameters)    

    # Architecture Search
    # parameters = architecture_search(path,best_parameters,possible_parameters)
    # parameters['experiment_Label'] = ['Risk_2_classes_de']    
    # parameters['experiment_Label'] = ['Risk_3_classes_died']#,'Osmonths','Vital status(Alive,Dead,OtherCauses)']    
    # parameters['experiment_Label'] = ['POLE Mutation','Copy number variation','MSI Status','Tumour Type']
    parameters['device'] = 'cuda:3'
    # parameters['SupervisedLearning_Lambda1'] = 0.000000001
    # parameters['batch_size'] = 16
    parameters['epochs'] = 40
    # parameters['showHowNetworkIsTraining']=True
    
    run_NaroNet(path,parameters)
    
    # Auto stats
    # get_BioInsights(path,parameters)

if __name__ == "__main__":
    # path = '/gpu-data/djsanchez/Images-SyntheticV1_v4/'  
    # path = '/gpu-data/djsanchez/Images-SyntheticV3_v4/'  
    # path = '/gpu-data/djsanchez/Images-SyntheticV2_v4/'    
    # path = '/gpu-data/djsanchez/Images-SyntheticV4_v4/' 
    # path = '/gpu-data/djsanchez/Images-SyntheticV_H1_v4/'
    # path = '/gpu-data/djsanchez/Images-SyntheticV_H2_v4/'
    # path = '/gpu-data/djsanchez/Images-SyntheticV_H3_v4/'
    # path = '/gpu-data/djsanchez/Images-LungCD8CD4FOXP3/'    
    # path = '/gpu-data/djsanchez/Images_LungQKBRCGLUT/'    
    # path = '/gpu-data/djsanchez/Images-Endometrial_LowGrade_NormRollBall_1Pat_v4/'
    # path = '/gpu-data/djsanchez/Images-ZuriBaselRisk_v4/'
    # path = '/gpu-data/djsanchez/Images-ZuriBaselRisk_1Im_Cell_v4/'
    # path = '/gpu-data/djsanchez/Images-ZuriBaselRisk_1Im_v4/'
    # path = '/gpu-data/djsanchez/Images-Endometrial_POLE_v4/'            
    # path = '/gpu-data/djsanchez/Images-KIRC/'    
    # path = '/gpu-data/djsanchez/Images-GBM/'    
    # path = '/gpu-data/djsanchez/Images-MouseBreast/'    
    path = '/gpu-data/djsanchez/Images-MouseBreast_10000cells/'    
    run(path)
 