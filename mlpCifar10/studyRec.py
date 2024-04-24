from maskGenerator import get_mask_pseudo_diagonal_numpy
import matplotlib.pyplot as plt
import numpy as np

# Function to create a mask with diagonals
def create_mask(start_rows, mask_shape):
    mask = np.zeros(mask_shape)
    diag_length = max(mask_shape)
    for start_row in start_rows:
        current_row, current_col = start_row % mask_shape[0], 0
        for _ in range(diag_length):
            mask[current_row % mask_shape[0], current_col % mask_shape[1]] = 1
            current_row += 1
            current_col += 1
    return mask

# Simulation parameters
diagonals_options = [2, 4, 8, 16, 32]
matrix_sizes = [50,100,150,200,250]
recurrence_counts = range(2, 21)

# Data structure to store results
results = {diag: {size: [] for size in matrix_sizes} for diag in diagonals_options}

from multiprocessing import Pool

# Define a function to encapsulate the per-diagonal-and-size computation
def compute_for_params(params):
    total_diag, mask_size = params
    mask_shape = [mask_size, mask_size]
    start_rows = np.random.choice(range(mask_shape[0]), total_diag, replace=False)
    initial_mask = create_mask(start_rows, mask_shape)
    
    # Store initial percentage of non-zeros
    non_zero_percent = np.count_nonzero(initial_mask) / initial_mask.size * 100
    percs = [non_zero_percent]
    
    # Perform matrix multiplications and track non-zero percentages
    current_matrix = initial_mask.copy()
    for _ in range(1, max(recurrence_counts)):
        current_matrix = np.dot(current_matrix, initial_mask)
        non_zero_percent = np.count_nonzero(current_matrix) / current_matrix.size * 100
        percs.append(non_zero_percent)
    
    return total_diag, mask_size, percs[:len(recurrence_counts)]

# Prepare the parameters for multiprocessing
param_list = [(total_diag, mask_size) for total_diag in diagonals_options for mask_size in matrix_sizes]

# Use a multiprocessing pool to compute in parallel
if __name__ == "__main__":
    with Pool(processes=16) as pool:  # Adjust the number of processes as needed
        results = pool.map(compute_for_params, param_list)

    # Process results to match the expected results structure
    processed_results = {diag: {size: [] for size in matrix_sizes} for diag in diagonals_options}
    for total_diag, mask_size, percs in results:
        processed_results[total_diag][mask_size] = np.array(percs)  # Convert the list to a numpy array

    # The rest of your plotting code remains the same


    # Plotting the results
    plt.figure(figsize=(20, 12), dpi=600)
    markers = ['o', 's', 'D', 'v', '^']
    for i, (diag_count, sizes) in enumerate(processed_results.items()):
        for j, (size, percentages) in enumerate(sizes.items()):
            plt.plot(recurrence_counts, percentages, marker=markers[j], label=f'Diag: {diag_count}, Size: {size}')
    plt.xlabel('Number of Recurrences')
    plt.ylabel('Percentage of Non-Zeros')
    plt.title('Matrix Multiplication Non-Zero Percentage by Recurrence')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('matrix_multiplication_non_zero_percentage.pdf')