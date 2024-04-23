#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.98 --diagPos1 128 146 175 206 208 --diagPos2 41 89 126 205 394
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.98 --diagPos1 55 90 122 144 159 --diagPos2 89 156 190 254 502
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.98 --diagPos1 8 34 162 213 247 --diagPos2 53 74 177 183 271
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.98 --diagPos1 170 178 192 224 225 --diagPos2 2 133 138 160 405
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.98 --diagPos1 90 116 138 251 252 --diagPos2 154 222 237 383 413
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.98 --diagPos1 35 36 68 150 234 --diagPos2 151 191 208 357 405
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.98 --diagPos1 24 99 121 162 183 --diagPos2 234 331 345 374 498
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.98 --diagPos1 14 63 73 123 194 --diagPos2 214 260 337 422 458
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.98 --diagPos1 75 100 153 161 196 --diagPos2 85 239 252 339 478
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.98 --diagPos1 46 56 58 64 239 --diagPos2 95 113 247 295 439
