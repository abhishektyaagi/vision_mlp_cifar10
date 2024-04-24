#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 1 --sparsity 0.99934895 --num_layers 1000 --diagPos 123 2116 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 2 --sparsity 0.99934895 --num_layers 1000 --diagPos 329 3051 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 3 --sparsity 0.99934895 --num_layers 1000 --diagPos 784 1003 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 4 --sparsity 0.99934895 --num_layers 1000 --diagPos 428 1076 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 5 --sparsity 0.99934895 --num_layers 1000 --diagPos 1750 2483 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 6 --sparsity 0.99934895 --num_layers 1000 --diagPos 32 546 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 7 --sparsity 0.99934895 --num_layers 1000 --diagPos 170 1824 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 8 --sparsity 0.99934895 --num_layers 1000 --diagPos 1399 2334 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 9 --sparsity 0.99934895 --num_layers 1000 --diagPos 1933 3006 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName 2DiagRec --net mlpmixer --n_epochs 300 --lr 1e-3 --expNum 10 --sparsity 0.99934895 --num_layers 1000 --diagPos 654 1406 
