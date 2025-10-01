from torch import nn


# Training configuration
training_config = {
    'num_epochs': 20000,
    'n_cpus': 10,
    'batch_size': 20000,
    # 'num_hidden_layers': 4,
    'activation': nn.Tanh(),
    'lossfn': nn.HuberLoss(delta=7),#nn.MSELoss(),
    'ensemble_size': 100,
    'bootstrap_counter_start': 1,
}

