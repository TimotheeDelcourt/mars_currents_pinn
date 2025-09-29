import torch.nn as nn
import numpy as np
import torch

class NeuralNet(nn.Module):
    def __init__(self,
                 num_hidden_layers = 9,
                 activation=nn.Tanh(),
                 ):
        super(NeuralNet, self).__init__()
        # Number of hidden layers 
        self.num_hidden_layers = num_hidden_layers
        # Number of neurons or units per layer 
        self.num_neurons = np.linspace(10,10+20*(num_hidden_layers-1),num_hidden_layers,dtype=int)
        # Activation function 
        self.activation = activation

        
        # create network by stacking layers
        self.input_layer = nn.Sequential(nn.Linear(4, self.num_neurons[0]),
                        self.activation)
        self.hidden_layers =[]
        for i in np.arange(1,num_hidden_layers):
            self.hidden_layers.append(nn.Linear(self.num_neurons[i-1], self.num_neurons[i]))
            self.hidden_layers.append(self.activation)
        self.output_layer = nn.Linear(self.num_neurons[-1], 3)
        self.network = nn.Sequential(*self.input_layer, *self.hidden_layers, self.output_layer)

    def forward(self, x):
        x = self.network(x)
        return x
    

