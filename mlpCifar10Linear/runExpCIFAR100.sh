#This script runs various dense versions of the MLPMixer on the CIFAR100 dataset
#Parameters at hand:
#Dim= 256,512
#Patchsize: 4, 8, 16
#depth:6,8,10
#An example command is: 
#reset && CUDA_VISIBLE_DEVICES=0 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 6 --dimhead 256

#All the possible combinations of command
#DENSE EXPERIMENT
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 6 --dimhead 512
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 8 --dimhead 256
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 8 --dimhead 512
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 10 --dimhead 256
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 10 --dimhead 512
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 8 --depth 6 --dimhead 256   
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 8 --depth 6 --dimhead 512
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 8 --depth 8 --dimhead 256   
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 8 --depth 8 --dimhead 512
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 8 --depth 10 --dimhead 256
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 8 --depth 10 --dimhead 512
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 16 --depth 6 --dimhead 256  
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 16 --depth 6 --dimhead 512
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 16 --depth 8 --dimhead 256  
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 16 --depth 8 --dimhead 512  
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 16 --depth 10 --dimhead 256 
#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 16 --depth 10 --dimhead 512

#reset && CUDA_VISIBLE_DEVICES=1 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 6 --dimhead 512
reset && CUDA_VISIBLE_DEVICES=0 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0 --num_layers 1 --k 0 --patch 4 --depth 8 --dimhead 1024
reset && CUDA_VISIBLE_DEVICES=0 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixerPar --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0.974 --num_layers 1 --k 25 --patch 4 --depth 8 --dimhead 1024
reset && CUDA_VISIBLE_DEVICES=0 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixerPar --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0.998 --num_layers 1 --k 12 --patch 4 --depth 8 --dimhead 1024
reset && CUDA_VISIBLE_DEVICES=0 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixerPar --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0.9483 --num_layers 1 --k 50 --patch 4 --depth 8 --dimhead 1024
reset && CUDA_VISIBLE_DEVICES=0 python trainCifar10PD.py --expName cifar100 --opt adamw --dataset cifar100 --net mlpmixerPar --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0.933 --num_layers 1 --k 64 --patch 4 --depth 8 --dimhead 1024 --resume


