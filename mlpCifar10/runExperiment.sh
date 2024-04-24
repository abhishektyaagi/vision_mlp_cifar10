reset && CUDA_VISIBLE_DEVICES=1  python train_cifar10.py --net mlpmixer  --num_layers 1
reset && CUDA_VISIBLE_DEVICES=1  python train_cifar10.py --net mlpmixer  --num_layers 2
reset && CUDA_VISIBLE_DEVICES=1  python train_cifar10.py --net mlpmixer  --num_layers 3
reset && CUDA_VISIBLE_DEVICES=1  python train_cifar10.py --net mlpmixer  --num_layers 4
reset && CUDA_VISIBLE_DEVICES=1  python train_cifar10.py --net mlpmixer  --num_layers 5