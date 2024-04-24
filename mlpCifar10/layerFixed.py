import os
import pdb
#Take the name of a file and read the contents into a list
def readFileToList(fileName):
    with open(fileName) as f:
        content = f.readlines()
    return content

acc2Diag= readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_layerFixed_0.99934895.txt")
acc4Diag = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_layerFixed_0.9986979.txt")

acc2DiagNotFixed = 39.28
acc4DiagNotFixed = 48.26
Dense = 56.76

#Arrange the lists above in descending order
acc2Diag.sort(reverse=True)
acc4Diag.sort(reverse=True)
#Have a list with values ranging from 0 to 99
#Convert the lists above from strings ending with \n to int
acc2Diag = [float(i.strip()) for i in acc2Diag]
acc4Diag = [float(i.strip()) for i in acc4Diag]
xAxis = [i for i in range(len(acc2Diag))]
#Plot the lists one one plot with legends
import matplotlib.pyplot as plt
import numpy as np
import pdb
plt.plot(xAxis, acc2Diag, label='2 Diag (0.99967)')
plt.plot(xAxis, acc4Diag, label='4 Diag (0.9986979)')
#A hornizontal line at accuracy of 56.76
plt.axhline(y=56.76, color='r', linestyle='--', label='Dense')
#A hornizontal line at accuracy of 39.28
plt.axhline(y=39.28, color='g', linestyle='--', label='2 Diag, No Fix')
#A hornizontal line at accuracy of 48.26
plt.axhline(y=48.26, color='b', linestyle='--', label='4 Diag, No Fix')
plt.xlabel('Configs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Configs')
plt.legend()
plt.savefig('accFixedOneLayer.png')