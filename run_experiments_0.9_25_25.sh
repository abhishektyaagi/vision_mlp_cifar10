#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 0 --sparsity 0.9 --diagPos1 31 32 48 55 61 65 72 75 76 91 92 98 129 147 154 155 160 161 162 179 197 211 232 244 248 --diagPos2 0 9 20 72 91 108 114 120 153 163 164 187 195 215 235 244 249 286 287 317 318 320 343 380 411
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 1 --sparsity 0.9 --diagPos1 4 18 23 26 33 43 57 64 68 80 87 96 106 136 154 157 158 167 175 187 207 230 236 241 248 --diagPos2 13 18 28 52 71 72 83 87 112 160 187 205 207 219 264 273 279 284 285 314 335 388 395 406 448
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 2 --sparsity 0.9 --diagPos1 9 15 20 23 26 27 37 46 48 68 81 93 116 126 148 153 166 167 174 181 184 189 193 206 239 --diagPos2 3 8 55 71 81 147 151 168 171 185 200 203 208 258 260 270 271 286 292 307 316 358 389 463 493
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 3 --sparsity 0.9 --diagPos1 2 3 29 31 36 40 43 61 72 78 88 97 103 116 127 137 147 149 164 169 172 181 191 202 244 --diagPos2 69 77 80 129 142 150 175 179 195 202 215 236 243 255 261 266 287 323 324 390 399 411 428 450 470
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 4 --sparsity 0.9 --diagPos1 10 15 39 63 65 73 77 86 92 94 98 119 133 136 175 177 189 195 198 210 218 221 223 226 254 --diagPos2 15 173 178 202 215 238 255 269 274 307 315 342 348 351 357 367 393 398 403 415 422 450 466 467 481
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 5 --sparsity 0.9 --diagPos1 0 5 7 32 48 58 87 91 92 94 109 128 142 149 153 173 183 189 224 225 231 232 234 238 247 --diagPos2 33 45 68 94 111 113 120 142 147 149 162 173 183 207 208 212 237 260 333 338 392 462 477 491 493
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 6 --sparsity 0.9 --diagPos1 0 16 23 33 37 78 89 94 95 114 124 139 150 151 156 182 195 197 198 199 201 203 209 242 249 --diagPos2 0 8 11 53 61 98 99 152 164 208 213 245 274 296 320 341 376 381 382 414 417 432 443 461 510
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 7 --sparsity 0.9 --diagPos1 0 1 7 13 24 30 46 53 56 67 91 98 121 125 127 137 149 156 163 169 181 183 207 217 219 --diagPos2 28 41 85 132 137 148 178 179 230 232 259 298 309 332 343 347 377 391 400 403 408 438 444 462 485
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 8 --sparsity 0.9 --diagPos1 3 5 7 14 18 20 36 39 45 48 51 81 101 108 118 120 133 162 166 175 188 198 231 232 249 --diagPos2 33 42 48 108 120 130 138 168 184 197 201 254 256 273 289 302 318 393 400 401 405 448 452 465 475
CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --expName rand --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum 9 --sparsity 0.9 --diagPos1 9 41 47 59 65 66 69 73 74 95 97 99 101 116 119 156 189 199 215 229 240 241 242 243 244 --diagPos2 4 21 24 35 62 76 89 90 118 122 123 195 198 237 267 272 326 343 370 404 432 443 462 502 508
