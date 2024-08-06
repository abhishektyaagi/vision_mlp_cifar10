from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from customFCGoogleSlow import CustomFullyConnectedLayer as customLinear
from customConv1dSlow import CustomConv1d
import pdb

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class permute(nn.Module):
    def __init__(self, dimension):
        super(permute, self).__init__()
        self.dimension = dimension


    def forward(self, x):
        if self.dimension == x.size()[-1]:
            return x
        else:
            return x.permute(0,2,1)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)


    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
""" def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, norm=512):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        permute(dim),
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
        permute(norm)
    ) """

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = customLinear,alphaLR=0.01,K=2,norm=512):
    inner_dim = int(dim * expansion_factor)
    #pdb.set_trace()
    return nn.Sequential(
        permute(dim),
        dense(dim, inner_dim,alphaLR=alphaLR,K=K),
        #dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim,alphaLR=alphaLR,K=K),
        #dense(inner_dim, dim),
        nn.Dropout(dropout),
        permute(norm)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 1,
              expansion_factor_token = 1, dropout = 0., alphaLR=0.01, K=2):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    #chan_first, chan_last = partial(nn.Linear), nn.Linear
    chan_first, chan_last = partial(customLinear), customLinear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first,norm = dim,alphaLR=alphaLR,K=K)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last,norm = dim,alphaLR=alphaLR,K=K))  
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
