import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pdb
import os
import argparse
from utils import progress_bar
from customFCGoogleSlow import CustomFullyConnectedLayer
import time

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,K=1,diagPos=[],alphaLR=0.01):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        
        self.relu = nn.ReLU()
        
        seed = 5
        torch.manual_seed(seed)
        #pdb.set_trace()
        self.recurrent_layer1 = CustomFullyConnectedLayer(input_dim, hidden_dim,K=K,diagPos=diagPos,alphaLR=alphaLR)
        #torch.manual_seed(seed)
        #self.recurrent_layer2 = nn.Linear(hidden_dim, hidden_dim)
        #pdb.set_trace()

        self.fc1 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        #x = self.relu(x)
        x = x.view(x.size(0), -1)

        x = self.recurrent_layer1(x)
        x = self.relu(x)
        x = self.fc1(x)
        
        return x