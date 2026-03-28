from torch import nn


# Training configuration
training_config = {
    'num_epochs': 20000,
    'n_cpus': 10,
    'batch_size': 20000,
    # 'num_hidden_layers': 4, NEURAL NET PARAMS IN neuralnets.py !!!
    'activation': nn.Tanh(),
    # 'lossfn': nn.HuberLoss(delta=7),#nn.MSELoss(),
    'lossfn': nn.MSELoss(),
    # 'ensemble_size': 100,
    'bootstrap_counter_start': 1,
    'bagging':False,
    'learning_rate': 0.1,
    'altitude_max':1500,
    'num_hidden_layers':1,
    'num_neurons_per_layer':8,
}




