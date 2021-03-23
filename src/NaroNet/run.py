
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

