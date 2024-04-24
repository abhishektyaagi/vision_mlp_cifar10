import torch
import torch.nn as nn
import numpy as np
from maskGenerator import get_mask_pseudo_diagonal_numpy
import pdb

def generate_corrected_random_mask(hidden_dim, input_dim, sparsity):
    """
    Generates a random mask of size hidden_dim x input_dim with 
    a total of (1-sparsity)*input_dim number of non-zero elements placed randomly across the entire matrix.
    
    Parameters:
    - hidden_dim (int): The number of rows in the mask.
    - input_dim (int): The number of columns in the mask.
    - sparsity (float): The fraction of elements that should be zero across the entire matrix.
    
    Returns:
    - np.array: The generated mask.
    """
    # Total number of elements in the matrix
    total_elements = hidden_dim * input_dim
    # Total number of non-zero elements in the matrix
    total_non_zeros = int((1 - sparsity) * input_dim*hidden_dim)
    
    # Initialize the mask with zeros
    mask = np.zeros(total_elements, dtype=np.float32)
    # Randomly choose indices for non-zero elements in the mask
    non_zero_indices = np.random.choice(total_elements, total_non_zeros, replace=False)
    # Set the chosen indices to 1
    mask[non_zero_indices] = 1
    # Reshape the mask back to the original dimensions
    mask = mask.reshape(hidden_dim, input_dim)
    
    return mask

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sparsity, diagPos = [], apply_mask=True):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        self.apply_mask = apply_mask
        
        self.relu = nn.ReLU()
        
        self.recurrent_layer1 = nn.Linear(input_dim, hidden_dim)
        #mask1 = generate_corrected_random_mask(hidden_dim, input_dim, sparsity)
        mask1 = get_mask_pseudo_diagonal_numpy((hidden_dim,input_dim), sparsity=sparsity, experimentType="randDiagOneLayer", layerNum=1, numDiag=len(diagPos)
                                                , diag_pos=diagPos, currLayer=1, debug=0)
        
        self.mask1 = nn.Parameter(torch.tensor(mask1, dtype=torch.float32),requires_grad=False)

        #Mask2 for the fc layer
        mask2 = get_mask_pseudo_diagonal_numpy((output_dim, hidden_dim), sparsity=sparsity, experimentType="randDiagOneLayer", layerNum=1, numDiag=len(diagPos)
                                                , diag_pos=[0], currLayer=1, debug=0)
        self.mask2 = nn.Parameter(torch.tensor(mask2, dtype=torch.float32),requires_grad=False)

        #self.recurrent_layer2 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

        if self.apply_mask:
            self.apply_mask_to_weights()
        
    def apply_mask_to_weights(self):
        #Apply the mask to the recurrent_layer1
        self.recurrent_layer1.weight.data *= self.mask1
        #Apply mask to the fc layer
        #self.fc1.weight.data *= self.mask2
        #self.recurrent_layer2.weight.data *= self.mask2
        
    def forward(self, x):
        #x = self.relu(x)
        x = x.view(x.size(0), -1)

        """ patches = torch.nn.functional.unfold(x, kernel_size=2, stride=2)
        num_patches = (32 // 2) * (32 // 2)

        # Reshape to have the patch dimension first, then channel, height, and width of the patch
        # Considering patches.shape[-1] gives us the total number of 2x2 patches across the image
        patches = patches.reshape(x.shape[0], 3, 2, 2, num_patches)

        # Permute to rearrange dimensions so we concatenate along the channel dimension last
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()

        # Finally, flatten the patches to get the desired ordering
        x = patches.view(x.shape[0], -1) """
        if self.apply_mask:
            self.apply_mask_to_weights()

        identity = x    
        for _ in range(self.num_layers):
            x = self.recurrent_layer1(x)
            x = self.relu(x)
            x = x + identity
            #x = nn.LayerNorm(x.size())(x)
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