import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from maskGenerator1Diag import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from torch_sparse_soft_topk_google.isotonic_dykstra import isotonic_dykstra_mask
from torch_sparse_soft_topk_google.topk import sparse_soft_topk_mask_dykstra
from torch_sparse_soft_topk_google.isotonic_pav import sparse_soft_topk_mask_pav

seed = 5
torch.manual_seed(seed)

class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=1, alphaLR=0.01):
        super(CustomConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K
        self.topkLR = alphaLR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.V = nn.Parameter(torch.empty(out_channels, out_channels, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(in_channels, device=self.device, requires_grad=True))
        nn.init.constant_(self.alpha, 1/in_channels)

        assert torch.all(self.alpha >= 0)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=self.device, dtype=torch.float32, requires_grad=True))
            fan_in = in_channels * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def compute_weights(self):
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()
        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)
        WSum = torch.zeros_like(self.weight)
        for i in non_zero_alpha_indices:
            mask = get_mask_pseudo_diagonal_torch(self.weight.shape, sparsity=0.99967, experimentType="randDiagOneLayer", diag_pos=i)
            #pdb.set_trace()
            #result = self.alpha_topk[i] * torch.einsum('ij,jkl->ikl', mask, self.V[i])
            #result = self.alpha_topk[i] * torch.matmul(mask, torch.diag(self.V[i]).to(self.device))
            V_i = torch.diag(self.V[i]).unsqueeze(2)  # Make V[i] into a 3D tensor to match mask
            result = self.alpha_topk[i] * mask * V_i
            WSum += result
        return WSum 

    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        #pdb.set_trace()
        W = self.weights
        return F.conv1d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR)