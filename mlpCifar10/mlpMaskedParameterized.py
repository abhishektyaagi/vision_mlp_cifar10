# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from customFCGoogleSlow import CustomFullyConnectedLayer as customLinear
from customConv1dSlow import CustomConv1d
import pdb

# Custom layer to print the shape of the weight matrix
class PrintWeightMatrixShape(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        print(f"Weight matrix shape: {self.layer.weight.shape}")
        return self.layer(x)

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        

    #Mask intended for layers in the model
    def forward(self, x):
        #pdb.set_trace()
        return self.fn(self.norm(x)) + x

class ForwardPrint(nn.Module):
    def __init__(self, prefix=""):
        super().__init__()
        self.prefix = prefix

    def forward(self, x):
        print(f"{self.prefix} shape: {x.shape}")
        return x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = customLinear,alphaLR=0.01,K=2):
    inner_dim = int(dim * expansion_factor)
    #pdb.set_trace()
    return nn.Sequential(
        dense(dim, inner_dim,alphaLR=alphaLR,K=K),
        #dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim,alphaLR=alphaLR,K=K),
        #dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 1,
              expansion_factor_token = 1, dropout = 0., alphaLR=0.01,K=2):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    #chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    chan_first, chan_last = partial(CustomConv1d,kernel_size= 1), customLinear
    #first_linear_layer = nn.Linear((patch_size ** 2) * channels, dim)

    #pdb.set_trace()
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        
        #NOTE:Can make these layers sparse
        #ForwardPrint(prefix="Input to first Linear layer"),
        #PrintWeightMatrixShape(first_linear_layer),
        #first_linear_layer,
        nn.Linear((patch_size ** 2) * channels, dim),
        #ForwardPrint(prefix="Output of first Linear layer"),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first,alphaLR=alphaLR,K=K)),
            #PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_first,alphaLR=alphaLR,K=K)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last,alphaLR=alphaLR,K=K))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        
        #NOTE:Can make these layers sparse
        nn.Linear(dim, num_classes)
    )
