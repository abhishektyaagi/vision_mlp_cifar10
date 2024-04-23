import itertools
import sys
import argparse
import random
import pdb
import math
import itertools

def generate_bash_script(experiment_name,sparsity):
    # Define the possible positions for the diagonals
    rows1 = 256
    cols1 = 512
    rows2 = 512
    cols2 = 256
    rows3 = 64
    cols3 = 256
    rows4 = 256
    cols4 = 64

    diagLength1 = max(rows1,cols1)
    diagLength2 = max(rows2,cols2)
    diagLength3 = max(rows3,cols3)
    diagLength4 = max(rows4,cols4)

    possible_positions1 = list(range(rows1))
    possible_positions2 = list(range(rows2))
    possible_positions3 = list(range(rows3))
    possible_positions4 = list(range(rows4))

    #pdb.set_trace()
    #calculate the sparsityL1 as a function of the number of diagonals
    num_diagonals1 = math.floor((1-sparsity)*(rows1*cols1)/diagLength1)
    num_diagonals2 = math.floor((1-sparsity)*(rows2*cols2)/diagLength2)
    num_diagonals3 = math.floor((1-sparsity)*(rows3*cols3)/diagLength3)
    num_diagonals4 = math.floor((1-sparsity)*(rows4*cols4)/diagLength4)
    
    """#Make a list of all possible combinations of diagonal positions
    listCombinations = list(itertools.combinations(possible_positions, num_diagonals))

    #Choose a certain number of diagonals from listCombinations randomly without repetition
    random_selection = random.sample(listCombinations, 200)"""

    random_selection1 = random_combinations(possible_positions1, num_diagonals1, 100)
    random_selection2 = random_combinations(possible_positions2, num_diagonals2, 100)
    random_selection3 = random_combinations(possible_positions3, num_diagonals3, 100)
    random_selection4 = random_combinations(possible_positions4, num_diagonals4, 100)

    file_name = f'run_experiments_{sparsity}_{num_diagonals1}_{num_diagonals2}_{num_diagonals3}_{num_diagonals4}.sh'
    expNum = 100
    # Open a bash script file to write the commands
    with open(file_name, 'w') as bash_script:
        bash_script.write('#!/bin/bash\n\n')

        # Generate all combinations of diagonal positions
        #for combination in itertools.combinations(possible_positions, num_diagonals):
        combinations1 = random_selection1
        combinations2 = random_selection2
        combinations3 = random_selection3
        combinations4 = random_selection4
        for i, (combination1, combination2, combination3, combination4) in enumerate(zip(combinations1, combinations2, combinations3, combinations4)):
            # Determine the CUDA_VISIBLE_DEVICES value
            #cuda_visible_devices = i // 4 % 2
            cuda_visible_devices = 0
            # Construct the command string
            command = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} python train_cifar10.py --expName {experiment_name} --net mlpmixer --n_epochs 500 --lr 1e-3 --expNum {expNum} --sparsity {sparsity} --diagPos1 {' '.join(map(str, combination1))} --diagPos2 {' '.join(map(str, combination2))} --diagPos3 {' '.join(map(str, combination3))} --diagPos4 {' '.join(map(str, combination4))} \n"
            bash_script.write(command)
            expNum += 1

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
    #parser.add_argument('a2', type=int, choices=range(1, 257), help='The number of diagonals (1-256)')
    
    args = parser.parse_args()
    
    # Generate the bash script
    generate_bash_script(args.experimentName, args.sparsity)

if __name__ == '__main__':
    main()
