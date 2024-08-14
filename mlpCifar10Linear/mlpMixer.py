from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from customFCGoogleSlow import CustomFullyConnectedLayer as customLinear
from customConv1dSlow import CustomConv1d
import pdb

#This is based on the code from here for cifar10: https://wandb.ai/sulbing/CIFAR10/reports/CIFAR10-Only-MLP-Not-CNN---Vmlldzo1NjkyNjMw

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
        #pdb.set_trace()
        return self.fn(self.norm(x)) + x
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, norm=512):
    inner_dim = int(dim * expansion_factor)
    #pdb.set_trace()
    return nn.Sequential(
        permute(dim),
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
        permute(norm)
    )

#All the commnets are for imgSize = 64, patchSize=8, dim=512, depth=8, num_classes=10
def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 1, expansion_factor_token = 1, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size) #num_patches = 64/8 * 64/8 = 64
    chan_first, chan_last = partial(nn.Linear), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  #Going from 512 3 64 64 to 512 64 192
        nn.Linear((patch_size ** 2) * channels, dim),   #Going from 192 to 512; output= [512 64 512]
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first,dim)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last,dim))
        ) for _ in range(depth)],
        #pdb.set_trace()
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
""" class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=1, expansion_factor_token=1, dropout=0.):
        super(MLPMixer, self).__init__()
        
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)  # num_patches = 64/8 * 64/8 = 64
        chan_first, chan_last = partial(nn.Linear), nn.Linear

        self.model = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),  # Going from [512, 3, 64, 64] to [512, 64, 192]
            nn.Linear((patch_size ** 2) * channels, dim),  # Going from 192 to 512; output= [512, 64, 512]
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first, dim)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last, dim))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        print("Input shape before Sequential block:", x.shape)
        return self.model(x) """