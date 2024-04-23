#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.95 --diagPos1 0 1 2 3 4 5 6 7 8 9 10 11 --diagPos2 0 1 2 3 4 5 6 7 8 9 10 11
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.95 --diagPos1 1 2 3 4 5 6 7 8 9 10 11 12 --diagPos2 1 2 3 4 5 6 7 8 9 10 11 12
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.95 --diagPos1 2 3 4 5 6 7 8 9 10 11 12 13 --diagPos2 2 3 4 5 6 7 8 9 10 11 12 13
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.95 --diagPos1 3 4 5 6 7 8 9 10 11 12 13 14 --diagPos2 3 4 5 6 7 8 9 10 11 12 13 14
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.95 --diagPos1 4 5 6 7 8 9 10 11 12 13 14 15 --diagPos2 4 5 6 7 8 9 10 11 12 13 14 15
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.95 --diagPos1 5 6 7 8 9 10 11 12 13 14 15 16 --diagPos2 5 6 7 8 9 10 11 12 13 14 15 16
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.95 --diagPos1 6 7 8 9 10 11 12 13 14 15 16 17 --diagPos2 6 7 8 9 10 11 12 13 14 15 16 17
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.95 --diagPos1 7 8 9 10 11 12 13 14 15 16 17 18 --diagPos2 7 8 9 10 11 12 13 14 15 16 17 18
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.95 --diagPos1 8 9 10 11 12 13 14 15 16 17 18 19 --diagPos2 8 9 10 11 12 13 14 15 16 17 18 19
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.95 --diagPos1 9 10 11 12 13 14 15 16 17 18 19 20 --diagPos2 9 10 11 12 13 14 15 16 17 18 19 20
