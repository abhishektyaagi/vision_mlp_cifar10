import matplotlib.pyplot as plt
import numpy as np
import pdb

denseAcc = 56.76
acc2Diag = 39.28
acc4Diag = 48.26
acc31Diag = 57.23

#Read values from a file into a list as integers
def read_values(filename):
    with open(filename) as f:
        try:
            return [x.strip() for x in f.readlines()]
        except ValueError:
            print("Error: Invalid literal found in the file.")
            return []

val2Diags = read_values("./log/log_mlpmixer_6_patch4_maxAcc_2DiagRec_0.99934895.txt")
val4Diags = read_values("./log/log_mlpmixer_6_patch4_maxAcc_4DiagRec_0.9986979.txt")
val31Diags = read_values("./log/log_mlpmixer_6_patch4_maxAcc_31DiagRec_0.99.txt")

#Convert the list of strings to a list of floats
val2Diags = [float(x) for x in val2Diags]
val4Diags = [float(x) for x in val4Diags]
val31Diags = [float(x) for x in val31Diags]

#Arrange the values in a descending order
val2Diags.sort(reverse=True)
val4Diags.sort(reverse=True)
val31Diags.sort(reverse=True)

#Plot the values
plt.figure(figsize=(10, 6))
plt.plot(val2Diags, label="2 Diags, N = 1000")
plt.plot(val4Diags, label="4 Diags, N = 30")
plt.plot(val31Diags, label="31 Diags, N = 5")
plt.axhline(y=denseAcc, color='r', linestyle='--', label="Dense Accuracy")
plt.axhline(y=acc2Diag, color='g', linestyle='--', label="2 Diags Accuracy")
plt.axhline(y=acc4Diag, color='b', linestyle='--', label="4 Diags Accuracy")
plt.axhline(y=acc31Diag, color='m', linestyle='--', label="31 Diags Accuracy")
plt.xlabel("Configs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Recurrences")
plt.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
plt.savefig("recvsdense.png")
