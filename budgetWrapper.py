#This module decides what is the amount of non-zeros for each layer 
from models.maskedmlpmixer import MLPMixer
import torch
import numpy as np
import pdb

#Get the model
net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = 4,
    dim = 512,
    depth = 6,
    num_classes = 10
    #diagPos1 = args.diagPos1,
    #diagPos2 = args.diagPos2,
    #sparsity=args.sparsity,
    #diagPos3 = args.diagPos3,
    #diagPos4 = args.diagPos4
)
#Load the model checkpoint stored in the path provided
checkpoint = torch.load('/p/dataset/abhishek/mlpmixerrand1_25-4-ckpt.t7')
net.load_state_dict(checkpoint['model'])
#Calculate the total number of parameters
total_params = sum(p.numel() for p in net.parameters())
#Print the dimensions of each layer in the model
for name, param in net.named_parameters():
    print(name, param.size())
#Calculate the number of non-zeros with given sparsity
sparsity = 0.9
num_non_zeros = total_params * (1-sparsity)
#Calculate the number of non-zeros for each layer based on its size
for name, param in net.named_parameters():
    if 'weight' in name:
        layer_size = param.size()
        if len(layer_size) == 2:
            num_non_zeros_layer = int(num_non_zeros * (layer_size[0] * layer_size[1]) / total_params)
            print(f"Number of non-zeros in {name}: {num_non_zeros_layer}")
        else:
            print(f"Skipping {name} as it is not a weight parameter")

#Calculate the number of diagonals for each layer by dividing the number of non-zeros by the max of two dimensions of the
#weight matrix and then using ceil function
num_diagonals_list = []
for name, param in net.named_parameters():
    if 'weight' in name:
        layer_size = param.size()
        if len(layer_size) == 2:
            num_non_zeros_layer = int(num_non_zeros * (layer_size[0] * layer_size[1]) / total_params)
            num_diagonals = int(np.ceil(num_non_zeros_layer / max(layer_size[0], layer_size[1])))
            print(f"Number of diagonals in {name}: {num_diagonals}")
        else:
            print(f"Skipping {name} as it is not a weight parameter")
        #Store the number of diagonals for each layer in a list
        num_diagonals_list.append(num_diagonals)

#Calculate the diagonal positions for each layer