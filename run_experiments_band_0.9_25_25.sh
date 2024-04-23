#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.9 --diagPos1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --diagPos2 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.9 --diagPos1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 --diagPos2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.9 --diagPos1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 --diagPos2 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.9 --diagPos1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 --diagPos2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.9 --diagPos1 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --diagPos2 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.9 --diagPos1 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --diagPos2 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.9 --diagPos1 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 --diagPos2 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.9 --diagPos1 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --diagPos2 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.9 --diagPos1 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 --diagPos2 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName band --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.9 --diagPos1 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 --diagPos2 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
