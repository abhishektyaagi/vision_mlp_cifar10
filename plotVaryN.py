import numpy as np
import matplotlib.pyplot as plt

# File names
file_names_set1 = [
    f"./log/log_mlpmixer_{i}_patch4_maxAcc_1_exp1_0.0.txt" for i in range(1, 7)
]

file_names_set2 = [
    f"./log/log_mlpmixerSquare_{i}_patch4_maxAcc_1_exp1_0.0.txt" for i in range(1, 7)
]

# List to store accuracies
accuracies_set1 = []
accuracies_set2 = []

# Read and store accuracies from files in the first set
for file_name in file_names_set1:
    with open(file_name, 'r') as file:
        accuracy = float(file.read().strip())
        accuracies_set1.append(accuracy)

# Read and store accuracies from files in the second set
for file_name in file_names_set2:
    with open(file_name, 'r') as file:
        accuracy = float(file.read().strip())
        accuracies_set2.append(accuracy)


#Plot the values on the same plot and save it
plt.plot(range(1, 7), accuracies_set1, label="MLP-Mixer")
plt.plot(range(1, 7), accuracies_set2, label="MLP-Mixer (Square)")
plt.xlabel("Number of Layers")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("varyN.png")