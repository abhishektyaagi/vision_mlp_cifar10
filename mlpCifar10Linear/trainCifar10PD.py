# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import deeplake
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from tin import TinyImageNetDataset
from tinImg import TinyImageNetHFDataset
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import os
import argparse
import pdb
import pandas as pd
import csv
import time
import math
import pickle

#from models import *
from utils import progress_bar
from randomaug import RandAugment
#from models.vit import ViT
#from models.convmixer import ConvMixer

from customFCGoogleSlow import CustomFullyConnectedLayer as customLinear
from customConv1dSlow import CustomConv1d

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_false', help='disable mixed precision training. for older pytorch versions')
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
#parser.add_argument('--diagPos2', nargs='+', default='0', metavar='P', help='diagonal position (default: 0)')
#parser.add_argument('--diagPos3', nargs='+', default='0', metavar='P', help='diagonal position (default: 0)')
#parser.add_argument('--diagPos4', nargs='+', default='0', metavar='P', help='diagonal position (default: 0)')
parser.add_argument('--expNum', type=int, default='1', metavar='E', help='experiment number (default: 1)')
parser.add_argument('--sparsity', default='0.8', type=float, help="sparsity for mask")
parser.add_argument('--depth', type=int, default='6', help='depth of transformer')
parser.add_argument('--num_layers', type=int, default='1', help='number of layers for MLP')
parser.add_argument("--k", type=int, default=1, help="number of neurons to keep")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')

args = parser.parse_args()
""" args = parser.parse_args()
k = math.floor((1-args.sparsity)*3072)
print("k: ", k) """
k = args.k
print(k)
alphaLR = 0.05
sparsity = args.sparsity

# Set the cache directory
os.environ["HF_DATASETS_CACHE"] = "/localdisk/Abhishek/.cache/huggingface/datasets"

# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = "{}_{}_lr{}_{}_d{}_p{}_dim{}".format(args.net,args.dataset, args.sparsity,k,args.depth,args.patch,args.dimhead)
    wandb.init(project="mlpParPD",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)
    imgSize = 32
    numClasses = 10
elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)
    imgSize = 32
    numClasses = 100
elif args.dataset == "tinyImageNet":
    print("Using tinyImageNet with imsize = 64")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    """ trainset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/tiny-imagenet-200/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=16)

    testset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/tiny-imagenet-200/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16) """

    """ trainset = TinyImageNetDataset(
        root_dir='../data/tinyData/tiny-imagenet-200',
        mode='train',
        preload=True,
        #load_transform=[transform_train],  #FIX: SEE IF THIS NEEDS FIXING FOR BETTER ACCURACY
        transform=transform_train,
        download=False,
        max_samples=None
    ) """
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=0)
    #trainld = deeplake.load("hub://activeloop/tiny-imagenet-train")
    #trainloader = trainld.pytorch(num_workers=0, batch_size=512, shuffle=False)

    """ testset = TinyImageNetDataset(
        root_dir='../data/tinyData/tiny-imagenet-200',
        mode='val',
        preload=True,
        #load_transform=[transform_test],
        transform=transform_test,
        download=False,
        max_samples=None
    ) """
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    #testld = deeplake.load("hub://activeloop/tiny-imagenet-test")
    #testloader = testld.pytorch(num_workers=0, batch_size=100, shuffle=False)

    # Load the datasets
    trainset = TinyImageNetHFDataset(split='train', transform=transform_train)
    testset = TinyImageNetHFDataset(split='valid', transform=transform_test)

    # Create the dataloaders
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    imgSize = 64
    numClasses = 200
elif args.dataset == "tinyImageNet224":
    trainset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/tiny-224/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=16)

    testset = torchvision.datasets.ImageFolder(root='../data/tiny-imagenet-200/tiny-224/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)
    imgSize = 64
    numClasses = 200

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
seed = 5
torch.manual_seed(seed)
# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    K = 0
    sparsity = 0
    from mlpMixer import MLPMixer
    net = MLPMixer(
            image_size = imgSize,
            channels = 3,
            patch_size = args.patch,
            dim = args.dimhead,
            depth = args.depth,
            num_classes = numClasses
        ).to(device) 
elif args.net=="mlpmixerPar":
    from mlpMixerPar import MLPMixer
    net = MLPMixer(
    image_size = imgSize,
    channels = 3,
    patch_size = args.patch,
    dim = args.dimhead,
    depth = args.depth,
    num_classes = numClasses ,
    alphaLR = alphaLR,
    K = args.k
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-{}-{}-ckpt.t7'.format(args.net,args.dataset,k))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

def get_cosine_annealed_lr(iteration, total_iterations, initial_lr, min_lr):
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * iteration / total_iterations))

if args.opt == "adam":
    #optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4) #Pixelated butterfly uses weight decay = 0.05 or 0.1 for ViT and Mixer
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

for name, param in net.named_parameters():
    print(name, param.size())
#pdb.set_trace()

#Print total number of trainable parameters
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total number of trainable parameters: ", total_params)
#pdb.set_trace()
##### Training
#Enable mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch,alphaLR = 0.01):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    #pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        #pdb.set_trace()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        
        #FIX THIS: save the checkpoint only after required K is achieved
        if epoch > 70:
            torch.save(state, f'checkpoint/mlpPD_{args.depth}_patch{args.patch}_{args.expName}_{sparsity}_{args.dataset}.pth')
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
net.cuda()
maxAcc = 0
alphaList = []
alphatopkList = []
alphaLR = get_cosine_annealed_lr(0, args.n_epochs, alphaLR, 1e-4)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch,alphaLR)
    val_loss, acc = test(epoch)

    #pdb.set_trace()
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    alphaLR = get_cosine_annealed_lr(epoch, args.n_epochs, alphaLR, 1e-4)
    
    # Log training..
    """ if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
                   "alphaLR": alphaLR,"Num_topk": len(torch.nonzero(net.recurrent_layer1.alpha_topk)),   
                    "epoch_time": time.time()-start}) """

    if usewandb:
        log_data = {
            'epoch': epoch, 
            'train_loss': trainloss, 
            'val_loss': val_loss, 
            'val_acc': acc, 
            'lr': optimizer.param_groups[0]["lr"],
            'alphaLR': alphaLR,
            'epoch_time': time.time() - start
        }

    #If a layer is customConv1d or customFC, update the alphaLR
    #NOTE: We know which modules are customConv1d and customFC. So we can use this to speed up the process
    numTopkDict = {}
    alphaDict = {}
    for name, module in net.named_modules():
        if isinstance(module, CustomConv1d) or isinstance(module, customLinear):
            #pdb.set_trace()
            print("Updating alphaLR for ", name)
            module.update_alpha_lr(alphaLR)
            num_topk = len(torch.nonzero(module.alpha_topk))
            numTopkDict[name] = num_topk
            log_data[f"Num_topk_{name}"] = num_topk
            alphaDict[name] = {
            'alphaList': [],
            'alphatopkList': []
            }

            if epoch % 5 == 0:
                # Extract alpha and alpha_topk
                if hasattr(module, 'alpha') and hasattr(module, 'alpha_topk'):
                    alpha = module.alpha.cpu().detach().numpy().flatten()
                    alphatopk = module.alpha_topk.cpu().detach().numpy()
                    
                    # Append the current values to the lists in the dictionary
                    alphaDict[name]['alphaList'].append(alpha)
                    alphaDict[name]['alphatopkList'].append(alphatopk)

    #pdb.set_trace()
    #net.recurrent_layer1.update_alpha_lr(alphaLR)

    #Save the maximum acc to a file
    if acc > maxAcc:
        maxAcc = acc


    #Get the value of alpha_topk from each layer
    """ for name, module in net.named_modules():
        if isinstance(module, CustomConv1d) or isinstance(module, customLinear):
            num_topk = len(torch.nonzero(module.alpha_topk))
            numTopkDict[name] = num_topk
            log_data[f"Num_topk_{name}"] = num_topk
            alphaDict[name] = {
            'alphaList': [],
            'alphatopkList': []
        } """
    
    wandb.log(log_data)

    """ if epoch % 5 == 0:
        #pdb.set_trace()
        alphatopk = net.recurrent_layer1.alpha_topk.cpu().detach().numpy()
        alpha = net.recurrent_layer1.alpha.cpu().detach().numpy().flatten()
        #alphatopksum = model.recurrent_layer1.alpha_topksum.cpu().numpy().flatten()
        alphaList.append(alpha)
        alphatopkList.append(alphatopk) """
    
    # During training, every 5 epochs, update the dictionary
    """ if epoch % 5 == 0:
        for name, module in net.named_modules():
            if isinstance(module, (CustomConv1d, customLinear)):
                # Extract alpha and alpha_topk
                if hasattr(module, 'alpha') and hasattr(module, 'alpha_topk'):
                    alpha = module.alpha.cpu().detach().numpy().flatten()
                    alphatopk = module.alpha_topk.cpu().detach().numpy()
                    
                    # Append the current values to the lists in the dictionary
                    alphaDict[name]['alphaList'].append(alpha)
                    alphaDict[name]['alphatopkList'].append(alphatopk)
 """
#Save maxAcc to a file
with open(f'log/log_{args.net}_{args.depth}_patch{args.patch}_maxAcc_{args.expName}_{args.sparsity}.txt', 'a') as f:
    f.write(str(maxAcc))
    #Put a new line character at the end
    f.write("\n")
    f.write(str(args.diagPos))
    f.write("\n")

#Save alpha to a file
save_path = f'./dataTopk/alpha_{args.k}.pkl'

# Save the dictionary to a file
with open(save_path, 'wb') as f:
    pickle.dump(alphaDict, f)

#np.save('./dataTopk/alphagoogle'+str(args.k)+'.npy', alphaList)
#np.save('./dataTopk/alphatopk'+str(args.k)+'.npy', alphatopkList)



