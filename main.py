from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.patch_contrastive_learning import patch_contrastive_learning
from NaroNet.Patch_Contrastive_Learning.preprocess_images import preprocess_images
from NaroNet.architecture_search.architecture_search import architecture_search
from NaroNet.NaroNet import run_NaroNet
from NaroNet.NaroNet_dataset import get_BioInsights

def main(path):
    # Select Experiment parameters
    params = parameters(path, 'Value')
    possible_params = parameters(path, 'Object')
    best_params = parameters(path, 'Index')    

    # Preprocess Images
    preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

    # Patch Contrastive Learning
    patch_contrastive_learning(path,params)    

    # Architecture Search
    # params = architecture_search(path,params,possible_params)

    # run_NaroNet(path,params)
    
    # BioInsights
    get_BioInsights(path,params)

if __name__ == "__main__":
    path = '/mnt/c/Users/danij/Google Drive/Proyectos/NaroNet/POLE_toy/'            
    main(path)
 