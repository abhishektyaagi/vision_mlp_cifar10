from mlpMasked import MLP
import argparse
import torch
import pdb
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='300')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--expName', default='exp1', type=str,help='experiment name')
parser.add_argument('--diagPos', nargs='+', default='0', metavar='P', help='diagonal position (default: 0)')
parser.add_argument('--diagPos2', nargs='+', default='0', metavar='P', help='diagonal position (default: 0)')
parser.add_argument('--diagPos3', nargs='+', default='0', metavar='P', help='diagonal position (default: 0)')
parser.add_argument('--diagPos4', nargs='+', default='0', metavar='P', help='diagonal position (default: 0)')
parser.add_argument('--expNum', type=int, default='1', metavar='E', help='experiment number (default: 1)')
parser.add_argument('--sparsity', default='0.8', type=float, help="sparsity for mask")
parser.add_argument('--depth', type=int, default='6', help='depth of transformer')
parser.add_argument('--num_layers', type=int, default='1', help='number of layers for MLP')

args = parser.parse_args()


net = MLP(
    input_dim = 3072,
    hidden_dim = 3072,
    output_dim = 10,
    num_layers=1,
    sparsity=args.sparsity,
    diagPos = args.diagPos
    #diagPos1 = args.diagPos1,
    #diagPos2 = args.diagPos2,
    #sparsity=args.sparsity,
    #diagPos3 = args.diagPos3,
    #diagPos4 = args.diagPos4
)

path4DiagRec = '/p/dataset/abhishek/mlpmixer4DiagRec11_4-4-ckpt.t7' 
#path2DiagRec = '/p/dataset/abhishek/mlpmixer2DiagRec1_2-4-ckpt.t7' 

# Load the weights for the model
checkpoint = torch.load(path4DiagRec)
net.load_state_dict(checkpoint['model'])

import matplotlib.pyplot as plt

#Calculate the number of non-zero weights in the recurrent_layer1
non_zero_weights = (net.recurrent_layer1.weight.data != 0).sum().item()
print(f'Number of non-zero weights in recurrent_layer1: {non_zero_weights}')
# Get the weights of the recurrent_layer1
recurrent_layer1_weights = net.recurrent_layer1.weight.data

# Set non-zero values to 1
recurrent_layer1_weights[recurrent_layer1_weights != 0] = 1
# Set zero values to 0
recurrent_layer1_weights[recurrent_layer1_weights == 0] = 0

# Plot the weights
plt.imshow(recurrent_layer1_weights, cmap='binary', vmin=0, vmax=1)
plt.title('Weights of recurrent_layer1')
plt.colorbar()
plt.savefig('./recurrent_layer1_weights_4DiagRec.png')
#plt.savefig('./recurrent_layer1_weights_2DiagRec.png')

#pdb.set_trace()
