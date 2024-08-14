# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from maskGenerator import get_mask_pseudo_diagonal_numpy
import pdb


pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, apply_mask=False, diagPos1=[], diagPos2=[], sparsity=0.8,diagPos3=[],diagPos4=[]):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.apply_mask = apply_mask

        #Fix this
        mask1 = get_mask_pseudo_diagonal_numpy((256,512), sparsity=sparsity, experimentType="randDiag", layerNum=1, numDiag=len(diagPos1)
                                                , diag_pos=diagPos1, currLayer=1, debug=0)
        self.mask1 = nn.Parameter(torch.tensor(mask1, dtype=torch.float32),requires_grad=False)

        mask2 = get_mask_pseudo_diagonal_numpy((512, 256), sparsity=sparsity, experimentType="randDiag", layerNum=1, numDiag=len(diagPos2)
                                                , diag_pos=diagPos2, currLayer=1, debug=0)
        self.mask2 = nn.Parameter(torch.tensor(mask2, dtype=torch.float32),requires_grad=False)

        mask3 = get_mask_pseudo_diagonal_numpy((64, 256), sparsity=sparsity, experimentType="randDiag", layerNum=1, numDiag=len(diagPos3)
                                                , diag_pos=diagPos3, currLayer=1, debug=0)
        mask3 = mask3.reshape(64, 256, 1)
        self.mask3 = nn.Parameter(torch.tensor(mask3, dtype=torch.float32),requires_grad=False)

        mask4 = get_mask_pseudo_diagonal_numpy((256, 64), sparsity=sparsity, experimentType="randDiag", layerNum=1, numDiag=len(diagPos4)
                                                , diag_pos=diagPos4, currLayer=1, debug=0)
        mask4 = mask4.reshape(256, 64, 1)
        self.mask4 = nn.Parameter(torch.tensor(mask4, dtype=torch.float32),requires_grad=False)

        if self.apply_mask:
            self.apply_mask_to_weights()

    def apply_mask_to_weights(self):
        # Apply the masks to the appropriate layers within the FeedForward block
        if isinstance(self.fn, nn.Sequential):
            layersLinear = [module for module in self.fn if isinstance(module, nn.Linear)]
            layersConv = [module for module in self.fn if isinstance(module, nn.Conv1d)]
            #pdb.set_trace()
            if len(layersLinear) > 0:
                layersLinear[0].weight.data *= self.mask1
            if len(layersLinear) > 1:
                layersLinear[1].weight.data *= self.mask2
            if len(layersConv) > 0:
                #pdb.set_trace()
                layersConv[0].weight.data *= self.mask4
            if len(layersConv) > 1:
                layersConv[1].weight.data *= self.mask3

    #Mask intended for layers in the model
    def forward(self, x):
        if self.apply_mask:
            self.apply_mask_to_weights()
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4,
              expansion_factor_token = 0.5, dropout = 0., diagPos1=[], diagPos2=[], sparsity=0.8, diagPos3=[], diagPos4=[]):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first), 
                            apply_mask=True, diagPos1=diagPos1, diagPos2 = diagPos2, sparsity=sparsity,diagPos3=diagPos3,diagPos4=diagPos4),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last),
                            apply_mask=True,diagPos1=diagPos1 ,diagPos2=diagPos2, sparsity=sparsity,diagPos3=diagPos3,diagPos4=diagPos4)
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
