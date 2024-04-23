#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.98 --diagPos1 0 55 99 140 173 --diagPos2 0 154 327 355 446
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.98 --diagPos1 0 55 88 95 199 --diagPos2 0 220 251 268 317
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.98 --diagPos1 0 37 81 105 255 --diagPos2 0 81 127 237 258
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.98 --diagPos1 0 48 69 104 122 --diagPos2 0 270 271 399 446
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.98 --diagPos1 0 133 139 157 183 --diagPos2 0 46 147 306 413
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.98 --diagPos1 0 65 83 161 190 --diagPos2 0 63 168 250 280
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.98 --diagPos1 0 13 22 125 169 --diagPos2 0 58 243 321 511
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.98 --diagPos1 0 119 130 160 185 --diagPos2 0 157 238 307 486
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.98 --diagPos1 0 62 95 158 188 --diagPos2 0 64 176 193 287
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.98 --diagPos1 0 132 183 242 254 --diagPos2 0 237 276 362 447
