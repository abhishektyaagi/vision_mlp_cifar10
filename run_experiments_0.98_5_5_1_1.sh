#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.98 --diagPos1 23 40 116 167 185 --diagPos2 44 232 311 332 400 --diagPos3 6 --diagPos4 37 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.98 --diagPos1 61 113 129 147 238 --diagPos2 120 230 307 311 476 --diagPos3 12 --diagPos4 230 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.98 --diagPos1 15 55 88 215 224 --diagPos2 20 285 291 382 465 --diagPos3 31 --diagPos4 229 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.98 --diagPos1 41 74 103 125 137 --diagPos2 207 352 410 462 488 --diagPos3 50 --diagPos4 226 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.98 --diagPos1 44 154 177 183 254 --diagPos2 43 48 87 289 406 --diagPos3 27 --diagPos4 71 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.98 --diagPos1 0 73 156 162 226 --diagPos2 370 378 449 475 510 --diagPos3 62 --diagPos4 0 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.98 --diagPos1 130 189 221 222 234 --diagPos2 28 164 253 289 505 --diagPos3 52 --diagPos4 74 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.98 --diagPos1 61 70 137 236 242 --diagPos2 183 212 384 406 485 --diagPos3 45 --diagPos4 125 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.98 --diagPos1 65 79 152 188 209 --diagPos2 21 40 84 386 423 --diagPos3 19 --diagPos4 19 
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.98 --diagPos1 22 85 123 163 244 --diagPos2 213 256 367 422 487 --diagPos3 25 --diagPos4 179 
