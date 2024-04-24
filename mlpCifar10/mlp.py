import torch

import torch.nn as nn
import pdb

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        
        self.relu = nn.ReLU()
        
        self.recurrent_layer1 = nn.Linear(input_dim, hidden_dim)
        #self.recurrent_layer2 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        #x = self.relu(x)
        x = x.view(x.size(0), -1)

        for _ in range(self.num_layers):
            x = self.recurrent_layer1(x)
            x = self.relu(x)
        """ for _ in range(self.num_layers):
            x = self.recurrent_layer2(x)
            x = self.relu(x)
         """
        x = self.fc1(x)
        
        return x


# Example usage
""" input_dim = 1024
hidden_dim = 1024
output_dim = 10
num_layers = 3

model = MLP(input_dim, hidden_dim, output_dim, num_layers) """