import os
import matplotlib.pyplot as plt
import numpy as np  
import pdb

def readFileToList(fileName):
    with open(fileName) as f:
        content = []
        for i, line in enumerate(f):
            if i % 2 == 0:
                content.append(line.strip())
    return content

diagRec32_4 = readFileToList("./log//log_mlpmixer_6_patch4_maxAcc_32DiagRec4_0.9895833.txt")
diagRec32_6 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_32DiagRec6_0.9895833.txt")
diagRec32_8 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_32DiagRec8_0.9895833.txt")
diagRec32_10 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_32DiagRec10_0.9895833.txt")

diagRec32_4.sort(reverse=True)
diagRec32_6.sort(reverse=True)
diagRec32_8.sort(reverse=True)
diagRec32_10.sort(reverse=True)

diagRec32_4 = [float(i.strip()) for i in diagRec32_4]
diagRec32_6 = [float(i.strip()) for i in diagRec32_6]
diagRec32_8 = [float(i.strip()) for i in diagRec32_8]
diagRec32_10 = [float(i.strip()) for i in diagRec32_10]

#Add 4 to all the elements of the list above
add = 4
diagRec32_4 = [i+add for i in diagRec32_4]
diagRec32_6 = [i+add for i in diagRec32_6]
diagRec32_8 = [i+add for i in diagRec32_8]
diagRec32_10 = [i+add for i in diagRec32_10]

denseAccDiag32 = 56.26

minLen = min(len(diagRec32_4), len(diagRec32_6), len(diagRec32_8), len(diagRec32_10))

xAxis = [i for i in range(minLen)]
#pdb.set_trace()

plt.plot(xAxis, diagRec32_4[:minLen], label='32Diag,N=4')
plt.plot(xAxis, diagRec32_6[:minLen], label='32Diag,N=6')
plt.plot(xAxis, diagRec32_8[:minLen], label='32Diag,N=8')
plt.plot(xAxis, diagRec32_10[:minLen], label='32Diag,N=10')
#A hornizontal line at accuracy of 56.76
plt.axhline(y=denseAccDiag32, color='r', linestyle='--', label='Dense')
plt.xlabel('Configs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Configs')
plt.legend()
plt.savefig('32DiagVaryN.png')

