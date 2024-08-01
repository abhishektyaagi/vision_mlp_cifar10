import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import numpy as np
import pdb
#from torch_sparse_soft_topk_google.isotonic_dykstra import isotonic_dykstra_mask
#from torch_sparse_soft_topk_google.topk import sparse_soft_topk_mask_dykstra

#sotonic_dykstra = types.SimpleNamespace()
#isotonic_dykstra.isotonic_dykstra_mask = isotonic_dykstra_mask

class CustomFullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, K=1, diagPos=[]):
        super(CustomFullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.num_permutations = in_features  # In the case of a square matrix
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the vectors V as learnable parameters
        self.V = nn.Parameter(torch.randn(out_features, out_features, device=self.device))

        # Initialize the alpha vector with 30 random values set to 1 and the rest to 0
        self.alpha = torch.zeros(self.num_permutations, device=self.device)
        """ indices = torch.randperm(self.num_permutations)
        self.alpha[indices[:30]] = 1 """

        #Set specific indices to 1 by going through the values in diagPos
        #pdb.set_trace()
        for i in diagPos:
            self.alpha[int(i)] = 1
        """  self.alpha[1014] = 1
        self.alpha[2160] = 1
        self.alpha[2411] = 1 """

    def sparse_permute_and_multiply(self, i, v_i):
        mask_shape = (self.out_features, self.out_features)
        diag_length = min(mask_shape)

        # Generate row and column indices
        rows = (torch.arange(diag_length, device=self.device) + i) % mask_shape[0]
        cols = torch.arange(diag_length, device=self.device) % mask_shape[1]

        # Create the sparse permutation matrix
        perm_indices = torch.stack([rows, cols])
        perm_values = torch.ones(diag_length, device=self.device)

        sparse_perm = torch.sparse_coo_tensor(perm_indices, perm_values, size=mask_shape, device=self.device)

        # Create the sparse diagonal matrix of v_i
        diag_indices = torch.stack([cols, cols])
        diag_values = v_i[cols]

        sparse_diag = torch.sparse_coo_tensor(diag_indices, diag_values, size=mask_shape, device=self.device)

        # Perform sparse matrix multiplication
        result = torch.sparse.mm(sparse_perm, sparse_diag)

        return result

    def compute_weights(self):
        self.alpha_topk = 0#sparse_soft_topk_mask_dykstra(self.alpha, self.K, 0.1, 50).to(self.device)

        # Find non-zero alpha indices
        non_zero_alpha_indices = torch.nonzero(self.alpha, as_tuple=False).squeeze()

        # Initialize the indices and values for the sparse weight matrix W
        all_indices = []
        all_values = []

        for i in non_zero_alpha_indices:
            permuted_product = self.sparse_permute_and_multiply(i, self.V[i])
            scaled_values = self.alpha[i] * permuted_product._values()
            all_indices.append(permuted_product._indices())
            all_values.append(scaled_values)

        # Combine all indices and values
        all_indices = torch.cat(all_indices, dim=1)
        all_values = torch.cat(all_values)

        # Create the final sparse tensor
        W_sparse = torch.sparse_coo_tensor(all_indices, all_values, size=(self.out_features, self.out_features), device=self.device)

        # Convert the final sparse W to a dense tensor
        W = W_sparse.to_dense()

        return W

    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        W = self.weights
        #Add assertion to check if the total non-zero values in the weight is equal to k*3072
        print(torch.sum(W != 0),self.K*3072)
        assert torch.sum(W != 0) == self.K*3072
        #If assetion fails, print the number of non-zeros
        
        out = F.linear(x, W)
        return out
