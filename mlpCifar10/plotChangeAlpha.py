import numpy as np
import matplotlib.pyplot as plt
import pdb

def get_non_zero_positions(vector):
    return np.nonzero(vector)[0]

# Load the list of vectors from the .npy file
vectors = np.load('./dataTopk/alphatopk2.npy')
vectorsAlpha4 = np.load('./dataTopk/alphagoogle2.npy')
#vectorsAlpha19 = np.load('./dataTopk/alphagoogle19.npy')
#vectorsAlpha20 = np.load('./dataTopk/alphagoogle20.npy')
pdb.set_trace()
# Get non-zero positions for all recorded vectors
non_zero_positions = [get_non_zero_positions(vec) for vec in vectors]

# Convert non-zero positions to binary matrix
binary_matrix = np.zeros((len(vectors), 3072))
for i, positions in enumerate(non_zero_positions):
    binary_matrix[i, positions] = 1

# Plot the binary matrix
plt.figure(figsize=(15, 8))
plt.imshow(binary_matrix, aspect='auto', cmap='gray')
plt.xlabel('Vector Index')
plt.ylabel('Epoch (every 5 epochs)')
plt.title('Non-Zero Positions Over Training')
plt.colorbar(label='Non-Zero Value')
plt.savefig('non_zero_positionsChange.png')

# Function to calculate Jaccard index
def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Calculate Jaccard index for consecutive epochs
jaccard_indices = []
for i in range(1, len(non_zero_positions)):
    set1 = set(non_zero_positions[i-1])
    set2 = set(non_zero_positions[i])
    jaccard_indices.append(jaccard_index(set1, set2))

pdb.set_trace()

# Plot Jaccard indices
jaccard_indicesDummy_lsmall = [0.12,0.84,0.74,0.54,0.41,0.69,0.84,0.21,0.36,0.45,0.54,0.23,0.52,0.81]
jaccard_indicesDummy_llarge = [0.99,0.99,1,1,1.00,1.00,1.0,1.00,1.00,1.00,1.00,1.00,1.00,1.00]
jaccard_indicesDummy_lsmall = jaccard_indices
plt.figure(figsize=(10, 5))
plt.plot(range(5, 5*len(jaccard_indicesDummy_lsmall)+1, 5), jaccard_indicesDummy_lsmall, marker='o',label='L = Adaptive')
#plt.plot(range(20, 20*len(jaccard_indicesDummy_llarge)+1, 20), jaccard_indicesDummy_llarge, marker='x',label='L = 0.1')
plt.xlabel('Epoch')
plt.ylabel('Jaccard Index')
plt.legend(loc='lower right')
plt.title('Jaccard Index of Non-Zero Positions Over Training')
plt.savefig('jaccardIndexChange.pdf')

def get_non_zero_values(vector):
    return vector[vector != 0]

# Get non-zero values for all recorded vectors
#non_zero_values19 = [get_non_zero_values(vec) for vec in vectorsAlpha19]
non_zero_values4 = [get_non_zero_values(vec) for vec in vectorsAlpha4]
mean_non_zero_values4 = [np.mean(values) for values in non_zero_values4]
std_non_zero_values4 = [np.std(values) for values in non_zero_values4]
# Calculate mean and standard deviation of non-zero values for each vector
#mean_non_zero_values19 = [np.mean(values) for values in non_zero_values19]
#std_non_zero_values19 = [np.std(values) for values in non_zero_values19]

#non_zero_values20 = [get_non_zero_values(vec) for vec in vectorsAlpha20]
# Calculate mean and standard deviation of non-zero values for each vector
#mean_non_zero_values20 = [np.mean(values) for values in non_zero_values20]
#std_non_zero_values20 = [np.std(values) for values in non_zero_values20]

# Plot mean and standard deviation of non-zero values over epochs
plt.figure(figsize=(10, 5))
plt.errorbar(range(20, 20*len(mean_non_zero_values4)+1, 20), mean_non_zero_values4, yerr=std_non_zero_values4, fmt='-o')
plt.xlabel('Epoch')
plt.ylabel('Non-Zero Values')
plt.title('Mean and Standard Deviation of Non-Zero Values Over Training')
plt.savefig('meanStdNonZeroValuesChange.png')

# Calculate mean absolute difference between consecutive epochs
mean_abs_diff19 = []
for i in range(1, len(non_zero_values4)):
    diff = np.abs(non_zero_values4[i] - non_zero_values4[i-1])
    mean_abs_diff19.append(np.mean(diff))

mean_abs_diff20 = []
for i in range(1, len(non_zero_values4)):
    diff = np.abs(non_zero_values4[i] - non_zero_values4[i-1])
    mean_abs_diff20.append(np.mean(diff))


# Plot mean absolute difference
plt.figure(figsize=(10, 5))
plt.plot(range(20, 20*len(mean_abs_diff19)+1, 20), mean_abs_diff19, marker='o')
plt.plot(range(20, 20*len(mean_abs_diff20)+1, 20), mean_abs_diff20, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Difference')
plt.title('Mean Absolute Difference of Non-Zero Values Over Training')
plt.savefig('meanAbsDiffChange.png')
