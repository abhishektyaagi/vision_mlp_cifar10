#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.98 --diagPos1 0 76 108 152 243 --diagPos2 0 235 346 383 400 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.98 --diagPos1 0 60 86 131 170 --diagPos2 0 164 329 374 467 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.98 --diagPos1 0 81 171 212 252 --diagPos2 0 227 347 412 422 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.98 --diagPos1 0 83 161 184 235 --diagPos2 0 114 152 456 477 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.98 --diagPos1 0 94 98 202 239 --diagPos2 0 103 169 301 490 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.98 --diagPos1 0 87 109 180 210 --diagPos2 0 349 448 463 496 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.98 --diagPos1 0 104 119 222 240 --diagPos2 0 325 346 446 509 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.98 --diagPos1 0 40 54 151 229 --diagPos2 0 334 335 393 475 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.98 --diagPos1 0 36 46 139 187 --diagPos2 0 390 421 449 487 --diagPos3 0 --diagPos4 0
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName atZero --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.98 --diagPos1 0 80 113 118 187 --diagPos2 0 230 235 340 355 --diagPos3 0 --diagPos4 0
