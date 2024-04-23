#!/bin/bash

python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.99609375 --diagPos1 186 --diagPos2 320
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.99609375 --diagPos1 6 --diagPos2 439
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.99609375 --diagPos1 2 --diagPos2 40
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.99609375 --diagPos1 240 --diagPos2 466
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.99609375 --diagPos1 139 --diagPos2 62
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.99609375 --diagPos1 123 --diagPos2 46
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.99609375 --diagPos1 71 --diagPos2 26
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.99609375 --diagPos1 129 --diagPos2 106
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.99609375 --diagPos1 254 --diagPos2 344
python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.99609375 --diagPos1 32 --diagPos2 212
