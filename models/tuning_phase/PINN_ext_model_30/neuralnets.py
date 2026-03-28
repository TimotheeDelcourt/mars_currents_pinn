import torch.nn as nn
import numpy as np
import torch

class NeuralNet(nn.Module):
    def __init__(self,
                num_hidden_layers,
                num_neurons_per_layer,
                activation,
                xyz_mean, # km
                xyz_std, # km
                alt_mean, # km
                alt_std, # km

                # all data:
                # xyz_mean = 6, # km
                # xyz_std = 3968, # km
                # alt_mean = 3235, # km
                # alt_std = 1829, # km

                # data < 175 km altitude:
                # xyz_mean = -202, # km
                # xyz_std = 2038, # km

                # data < 250 km altitude:
                # xyz_mean = -48,
                # xyz_std = 2069,
                # alt_mean = 195, # km
                # alt_std = 31, # km

                # data < 600 km altitude:
                # xyz_mean = 18,
                # xyz_std = 2144,
                # alt_mean = 321, # km
                # alt_std = 131, # km

                # sin_colat_mean = 0.7,
                # sin_colat_std = 0.3,
                # cos_colat_mean = 0.0,
                # cos_colat_std = 0.7,
                # sin_lon_mean = 0.0,
                # sin_lon_std = 0.7,
                # cos_lon_mean = 0.0,
                # cos_lon_std = 0.7,
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
        self.xyz_mean = xyz_mean
        self.xyz_std = xyz_std
        self.alt_mean = alt_mean
        self.alt_std = alt_std
        # self.sin_colat_mean = sin_colat_mean
        # self.sin_colat_std = sin_colat_std
        # self.cos_colat_mean = cos_colat_mean
        # self.cos_colat_std = cos_colat_std
        # self.sin_lon_mean = sin_lon_mean
        # self.sin_lon_std = sin_lon_std
        # self.cos_lon_mean = cos_lon_mean
        # self.cos_lon_std = cos_lon_std

        # try:
        #     DEVICE = torch.device('privateuseone:0')
        # except:
        #     if torch.cuda.is_available():
        #         DEVICE = torch.device('cuda')
        #     else:
        #         DEVICE = 'cpu'

        # self.means = torch.tensor([xyz_mean, xyz_mean, xyz_mean,
        #                            alt_mean, sin_colat_mean, cos_colat_mean,
        #                            sin_lon_mean, cos_lon_mean], device=DEVICE)

        # self.stds = torch.tensor([xyz_std, xyz_std, xyz_std,
        #                            alt_std, sin_colat_std, cos_colat_std,
        #                            sin_lon_std, cos_lon_std], device=DEVICE)

        # create network by stacking layers
        if self.num_hidden_layers > 1:
            self.input_layer = nn.Sequential(nn.Linear(4, self.num_neurons[0]),
                            self.activation)
            self.hidden_layers =[]
            for i in np.arange(1,num_hidden_layers):
                self.hidden_layers.append(nn.Linear(self.num_neurons[i-1], self.num_neurons[i]))
                self.hidden_layers.append(self.activation)
            self.output_layer = nn.Linear(self.num_neurons[-1], 3)
            self.network = nn.Sequential(*self.input_layer, *self.hidden_layers, self.output_layer)

        elif self.num_hidden_layers == 1:
            self.input_layer = nn.Sequential(nn.Linear(4, self.num_neurons[0]),
                            self.activation)
            self.output_layer = nn.Linear(self.num_neurons[0], 3)
            self.network = nn.Sequential(*self.input_layer, self.output_layer)

    def forward(self, x):
        x_norm = (x[:, :3] - self.xyz_mean) / self.xyz_std
        alt_norm = (x[:, 3] - self.alt_mean) / self.alt_std
        x_norm = torch.cat([x_norm, alt_norm.unsqueeze(1)], dim=1)
        x = self.network(x_norm)
        return x
    

class NeuralNet_indep(nn.Module):
    def __init__(self,
                num_hidden_layers,
                num_neurons_per_layer,
                activation,
                xyz_mean, # km
                xyz_std, # km
                ):
        super(NeuralNet_indep, self).__init__()
        # Number of hidden layers 
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = np.ones(num_hidden_layers, dtype=int)*num_neurons_per_layer
        # Activation function 
        self.activation = activation
        # Standardization parameters
        self.xyz_mean = xyz_mean
        self.xyz_std = xyz_std

        self.network_x = self._build_network()
        self.network_y = self._build_network()
        self.network_z = self._build_network()

    def _build_network(self):
        if self.num_hidden_layers == 1:
            self.input_layer = nn.Sequential(nn.Linear(3, self.num_neurons[0]),
                            self.activation)
            self.output_layer = nn.Linear(self.num_neurons[0], 1)
            return nn.Sequential(*self.input_layer, self.output_layer)
        else:
            self.input_layer = nn.Sequential(nn.Linear(3, self.num_neurons[0]),
                        self.activation)
            self.hidden_layers =[]
            for i in np.arange(1,self.num_hidden_layers):
                self.hidden_layers.append(nn.Linear(self.num_neurons[i-1], self.num_neurons[i]))
                self.hidden_layers.append(self.activation)
            self.output_layer = nn.Linear(self.num_neurons[-1], 1)
            return nn.Sequential(*self.input_layer, *self.hidden_layers, self.output_layer)

    def forward(self, x):
        x_norm = (x - self.xyz_mean) / self.xyz_std
        
        x_out = self.network_x(x_norm)
        y_out = self.network_y(x_norm)
        z_out = self.network_z(x_norm)

        output = torch.cat([x_out, y_out, z_out], dim=1)

        return output
    


