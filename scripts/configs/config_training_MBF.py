from torch import nn
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Training configuration
training_config = {
    'num_epochs': 20000,
    'n_cpus': 10,
    'batch_size': 20000,
    'activation': nn.Tanh(),
    'lossfn': nn.MSELoss(),
    'ensemble_size': 60,
    'bootstrap_counter_start': 1,
    'validation':True,
    'learning_rate': 0.1,
    'num_hidden_layers':1,
    'random_parameters':False, # NOT including alt as input
    'sample_with_replacement':False,
    'add_folder_str':'', # leave '' if nothing to add to the folder name

    # pre-determined parameters
    'altitude_max':650,
    'num_neurons_per_layer':25,
    # 'l1_lambda':0,
    'include_alt':False,
    # 'smoothness_lambda':1e-6,

    # randomly selected parameters
    # 'l1_lambdas':[0, 0, 1e-7,1e-6,5*1e-6],
    'altitudes_max':[220,650],# can contain a single value, or start and stop values for random selection
    'nums_neurons_per_layer':[18,30],
    'crop_outlier':[30,40,50,60,70], # can be a list or contain a single value

}

