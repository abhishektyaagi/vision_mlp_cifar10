import torch
import torch.nn as nn
import numpy as np
import pdb
#from mlp import MLP as mlp
from mlpGoogleSlow import MLP as mlp
#from mlpPicka import MLP as mlpPicka
import matplotlib.pyplot as plt

#This script loads the individual mlps and then plots the heatmap of the weight
givenK = 2
mlpModel = mlp(input_dim=3072, hidden_dim=3072, output_dim=10, num_layers=1,K=givenK)

#Load the checkpoint from the directory
checkpoint = torch.load("./checkpoint/mlpPD_6_patch4_diffDiag_0.9993489583333334.pth")

#Load the weights from directory checkpoints
#mlpModel.load_state_dict(torch.load("./checkpoints/checkpoint_"+str(givenK)+".pth"))
model_state_dict = checkpoint['model']
mlpModel.load_state_dict(model_state_dict)

#Get the weights from the layer called recurrent_layer1
weights = mlpModel.recurrent_layer1.weights

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
plt.title('Weight Matrix with K=3 and topkGoogle')
plt.savefig('heatmapWeightsGoogle' + str(givenK) + '.png')
plt.savefig('heatmapWeightsGoogle' + str(givenK) + '.pdf')
plt.close()

#Output the actual number of non-zeros in the weights
print("Number of non-zero values in weightsGoogle: ", torch.count_nonzero(weights).item())
print("Ground truth number of non-zero values in weightsGoogle: ", givenK*3072)
print("Obtained number of diagonals: ", torch.count_nonzero(weights).item()/3072)
print("Ground truth number of diagonals: ", givenK)

#Write the difference between the two numbers as errors
error = abs(torch.count_nonzero(weights).item() - givenK*3072)
print("Error in the number of non-zero values: ", error)

print('---------------------------------')
#Do the same thing as above but for eps weights
print("Number of non-zero values in weightsEps: ", torch.count_nonzero(weights).item())
print("Ground truth number of non-zero values in weightsEps: ", givenK*3072)
print("Obtained number of diagonals: ", torch.count_nonzero(weights).item()/3072)
print("Ground truth number of diagonals: ", givenK)

#Write the difference between the two numbers as errors
error = abs(torch.count_nonzero(weights).item() - givenK*3072)
print("Error in the number of non-zero values: ", error)

