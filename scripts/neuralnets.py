import torch.nn as nn
import numpy as np
import torch

class NeuralNet(nn.Module):
    def __init__(self,
                num_hidden_layers=1,
                num_neurons_per_layer=32,
                activation=nn.Tanh(),
                xyz_mean = 6, # km
                xyz_std = 3968, # km
                alt_mean = 3235, # km
                alt_std = 1829, # km
                sin_colat_mean = 0.7,
                sin_colat_std = 0.3,
                cos_colat_mean = 0.0,
                cos_colat_std = 0.7,
                sin_lon_mean = 0.0,
                sin_lon_std = 0.7,
                cos_lon_mean = 0.0,
                cos_lon_std = 0.7,
                ):
        super(NeuralNet, self).__init__()
        # Number of hidden layers 
        self.num_hidden_layers = num_hidden_layers
        # Number of neurons or units per layer 
        # self.num_neurons = np.linspace(10,10+20*(num_hidden_layers-1),num_hidden_layers,dtype=int)
        self.num_neurons = np.ones(num_hidden_layers, dtype=int)*num_neurons_per_layer
        # Activation function 
        self.activation = activation
        # Standardization parameters
        # self.xyz_mean = xyz_mean
        # self.xyz_std = xyz_std
        # self.alt_mean = alt_mean
        # self.alt_std = alt_std
        # self.sin_colat_mean = sin_colat_mean
        # self.sin_colat_std = sin_colat_std
        # self.cos_colat_mean = cos_colat_mean
        # self.cos_colat_std = cos_colat_std
        # self.sin_lon_mean = sin_lon_mean
        # self.sin_lon_std = sin_lon_std
        # self.cos_lon_mean = cos_lon_mean
        # self.cos_lon_std = cos_lon_std

        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = 'cpu'

        self.means = torch.tensor([xyz_mean, xyz_mean, xyz_mean,
                                   alt_mean, sin_colat_mean, cos_colat_mean,
                                   sin_lon_mean, cos_lon_mean]).to(DEVICE)

        self.stds = torch.tensor([xyz_std, xyz_std, xyz_std,
                                   alt_std, sin_colat_std, cos_colat_std,
                                   sin_lon_std, cos_lon_std]).to(DEVICE)

        # create network by stacking layers
        if self.num_hidden_layers > 1:
            self.input_layer = nn.Sequential(nn.Linear(8, self.num_neurons[0]),
                            self.activation)
            self.hidden_layers =[]
            for i in np.arange(1,num_hidden_layers):
                self.hidden_layers.append(nn.Linear(self.num_neurons[i-1], self.num_neurons[i]))
                self.hidden_layers.append(self.activation)
            self.output_layer = nn.Linear(self.num_neurons[-1], 3)
            self.network = nn.Sequential(*self.input_layer, *self.hidden_layers, self.output_layer)

        elif self.num_hidden_layers == 1:
            self.input_layer = nn.Sequential(nn.Linear(8, self.num_neurons[0]),
                            self.activation)
            self.output_layer = nn.Linear(self.num_neurons[0], 3)
            self.network = nn.Sequential(*self.input_layer, self.output_layer)

    def forward(self, x):
        # x_norm = (x[:, :3] - self.xyz_mean) / self.xyz_std
        # alt_norm = (x[:, 3] - self.alt_mean) / self.alt_std
        # x_norm = torch.cat([x_norm, alt_norm.unsqueeze(1)], dim=1)
        x_norm = (x - self.means) / self.stds
        x = self.network(x_norm)
        return x
    

