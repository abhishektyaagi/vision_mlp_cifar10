import os
import random
# Read accuracies from file
import matplotlib.pyplot as plt

# Get the list of files in the log directory
log_dir = './log'
files = os.listdir(log_dir)

# Read accuracies from files
accuracies = []
for file in files:
    if file.startswith('log_mlpmixer_patch4_maxAcc_') and file.endswith('_rand_0.9.txt'):
        file_path = os.path.join(log_dir, file)
        with open(file_path, 'r') as f:
            accuracy = float(f.read().strip())
            accuracies.append(accuracy)

# Sort accuracies in descending order
accuracies.sort(reverse=True)

# Plot scatter plot
plt.scatter(range(len(accuracies)), accuracies, label='Sparse')

# Add horizontal dashed line
plt.axhline(y=86.76, color='r', linestyle='--', label='Dense')

# Set plot labels and title
plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.ylim(60,90)
# Set plot title
plt.title('Randomly Chosen configurations with 90% sparsity\n(FC1: 6, MLP1: 25, MLP2: 25, FC2: 6)')

# Show legend
plt.legend()

# Show the plot
plt.savefig("acc0.9.png")
plt.close() 

accuraciesRand = random.sample(accuracies, 10)
accuraciesRand.sort(reverse=True)

accuraciesBand = [83.37,81.22,81.7,80.86,83.59,82.01,80.55,80.44,81.97,82.18]
accuraciesBand.sort(reverse=True)

accuraciesSmallBand = [84.21,82.29,82.65,83.67,82.54,84.06,83.61,83.54,82.36]
accuraciesSmallBand.sort(reverse=True)

#Plot all the three lists on the same graph
plt.plot(range(len(accuraciesRand)), accuraciesRand, label='Random')
plt.plot(range(len(accuraciesBand)), accuraciesBand, label='Band')
plt.plot(range(len(accuraciesSmallBand)), accuraciesSmallBand, label='Small Band')
plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(60,90)
plt.title('Comparison of Random, Band and Small Band configurations at 90% sparsity\n(FC1: 6, MLP1: 25, MLP2: 25, FC2: 6)')
plt.savefig("acc0.9_compare.png")
plt.close()




