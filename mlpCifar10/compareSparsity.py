import os
#Take the name of a file and read the contents into a list
def readFileToList(fileName):
    with open(fileName) as f:
        content = f.readlines()
    return content

DiagList1 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_1Diag_0.999674479.txt")
DiagList31 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_31Diag_0.99.txt")
DiagList2 = readFileToList("/p/dataset/abhishek/log_mlpmixer_6_patch4_maxAcc_2Diag_0.999348958.txt")

#Arrange the lists above in descending order
DiagList1.sort(reverse=True)
DiagList31.sort(reverse=True)
DiagList2.sort(reverse=True)
#Have a list with values ranging from 0 to 99
#Convert the lists above from strings ending with \n to int
DiagList1 = [float(i.strip()) for i in DiagList1]
DiagList31 = [float(i.strip()) for i in DiagList31]
DiagList2 = [float(i.strip()) for i in DiagList2]

xAxis = [i for i in range(100)]

#Plot the lists one one plot with legends
import matplotlib.pyplot as plt
import numpy as np
import pdb
pdb.set_trace()
plt.plot(xAxis, DiagList1, label='1 Diag (0.99967)')
plt.plot(xAxis, DiagList2, label='2 Diag (0.99934)')
plt.plot(xAxis, DiagList31, label='31 Diag (0.99)')
#A hornizontal line at accuracy of 56.76
plt.axhline(y=56.76, color='r', linestyle='--', label='Dense')
plt.xlabel('Configs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Configs')
plt.legend()
plt.savefig('compareSparsity.png')