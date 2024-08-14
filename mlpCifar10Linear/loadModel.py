import torch
import torch.nn as nn
import numpy as np
import pdb
from mlpMixerPar import MLPMixer as mlp
import matplotlib.pyplot as plt
import pdb

#This script loads the individual mlps and then plots the heatmap of the weight
givenK = 2
#mlpModel = mlp(input_dim=3072, hidden_dim=3072, output_dim=10, num_layers=1,K=givenK)
mlpModel = mlp(
    image_size = 32,
    channels = 3,
    patch_size = 4,
    dim = 256,
    depth = 6,
    num_classes = 10,
    alphaLR = 0.1,
    K = 25
)

#Load the checkpoint from the directory
checkpoint = torch.load("./checkpoint/mlpPD_6_patch4_diffDiag_0.9993489583333334.pth")

#Load the weights from directory checkpoints
#mlpModel.load_state_dict(torch.load("./checkpoints/checkpoint_"+str(givenK)+".pth"))
model_state_dict = checkpoint['model']
mlpModel.load_state_dict(model_state_dict)

#Calculate the total number of parameters in the model
total_params = sum(p.numel() for p in mlpModel.parameters())
print("Total number of parameters in the model: ", total_params)

#Get the total number of non-zeros in the model
total_non_zeros = sum(p.nonzero().size(0) for p in mlpModel.parameters())
print("Total number of non-zeros in the model: ", total_non_zeros)
pdb.set_trace()
#weights = mlpModel.recurrent_layer1.weight.data
weights = mlpModel[2][1].fn[1].weights.data

#Get the indices of non-zero values
indices = torch.nonzero(weights)

#Save it to a file
with open('nonZeroValuesGoogle' + str(givenK) + '.txt', 'w') as file:
    for i in indices:
        file.write(str(i) + '\n')

#Take the above indices and put a 1 at the indices above and zero in reset of the indices for a 3072x3072 matrix
#Create a 3072x3072 matrix with zeros
heatmap_matrix = np.zeros(weights.shape)

# Set the color for the specified indices
for idx in indices:
    i, j = idx
    heatmap_matrix[i, j] = 1

# Plot the heatmap
plt.figure(figsize=[10, 10])
plt.imshow(heatmap_matrix, cmap='hot', interpolation='nearest', aspect='auto')
#plt.colorbar()
plt.title('Weight Matrix')
plt.savefig('heatmapWeights' + str(givenK) + '.png')
plt.savefig('heatmapWeights' + str(givenK) + '.pdf')
plt.close()
