import itertools
import sys
import argparse
import random
import pdb
import math
import itertools

def generate_bash_script(experiment_name,sparsity, num_layers):
    # Define the possible positions for the diagonals
    rows1 = 3072
    cols1 = 3072
    
    diagLength1 = max(rows1,cols1)

    possible_positions1 = list(range(rows1))
    #pdb.set_trace()

    #pdb.set_trace()
    #calculate the sparsityL1 as a function of the number of diagonals
    num_diagonals1 = 1

    #pdb.set_trace()
    file_name = f'run_experiments_1Diag_1.sh'
    expNum = 42

    # Open a bash script file to write the commands
    with open(file_name, 'w') as bash_script:
        bash_script.write('#!/bin/bash\n\n')

        # Generate all combinations of diagonal positions
        #for combination in itertools.combinations(possible_positions, num_diagonals):
        i = 0
        for combination1 in range(rows1):
            # Determine the CUDA_VISIBLE_DEVICES value
            cuda_visible_devices = i // 6 % 2
            #pdb.set_trace()
            #cuda_visible_devices = 0,1
            # Construct the command string
            command = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}  python train_cifar10.py --expName {experiment_name} --net mlpmixer --n_epochs 200 --lr 1e-3 --expNum {expNum} --sparsity 0.99967 --num_layers 1 --diagPos {combination1} \n"
            bash_script.write(command)
            #expNum += 1
            i += 1

def generate_random_combination(iterable, r):
    """Generate a random combination of r elements from the iterable."""
    items = list(iterable)
    random_combination = random.sample(items, r)
    return tuple(sorted(random_combination))

def random_combinations(iterable, r, num_samples):
    """Efficiently generate a set of unique random combinations."""
    combinations_set = set()
    while len(combinations_set) < num_samples:
        combination = generate_random_combination(iterable, r)
        combinations_set.add(combination)
    return list(combinations_set)
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a bash script for running experiments with different diagonal positions.')
    parser.add_argument('experimentName', type=str, help='The name of the experiment')
    parser.add_argument('sparsity', type=float, help='The amount of sparsity in the network')
    parser.add_argument('num_layers', type=int, help='The number of layers in the network')
    #parser.add_argument('a2', type=int, choices=range(1, 257), help='The number of diagonals (1-256)')
    
    args = parser.parse_args()
    
    # Generate the bash script
    generate_bash_script(args.experimentName, args.sparsity, args.num_layers)

if __name__ == '__main__':
    main()
