#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.98 --diagPos1 0 1 2 3 4 --diagPos2 0 1 2 3 4 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.98 --diagPos1 1 2 3 4 5 --diagPos2 1 2 3 4 5 --diagPos3 1 --diagPos4 1
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.98 --diagPos1 2 3 4 5 6 --diagPos2 2 3 4 5 6 --diagPos3 2 --diagPos4 2
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.98 --diagPos1 3 4 5 6 7 --diagPos2 3 4 5 6 7 --diagPos3 3 --diagPos4 3
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.98 --diagPos1 4 5 6 7 8 --diagPos2 4 5 6 7 8 --diagPos3 4 --diagPos4 4
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.98 --diagPos1 5 6 7 8 9 --diagPos2 5 6 7 8 9 --diagPos3 5 --diagPos4 5
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.98 --diagPos1 6 7 8 9 10 --diagPos2 6 7 8 9 10 --diagPos3 6 --diagPos4 6
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.98 --diagPos1 7 8 9 10 11 --diagPos2 7 8 9 10 11 --diagPos3 7 --diagPos4 7
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.98 --diagPos1 8 9 10 11 12 --diagPos2 8 9 10 11 12 --diagPos3 8 --diagPos4 8
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.98 --diagPos1 9 10 11 12 13 --diagPos2 9 10 11 12 13 --diagPos3 9 --diagPos4 9
