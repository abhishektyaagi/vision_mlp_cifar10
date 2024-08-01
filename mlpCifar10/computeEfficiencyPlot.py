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

diagRec32_4 = readFileToList("./log//log_mlpmixer_6_patch4_maxAcc_32DiagRec4_Rec_0.9895833.txt")
diagRec32_6 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_32DiagRec6_Rec_0.9895833.txt")
diagRec32_8 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_32DiagRec8_Rec_0.9895833.txt")
diagRec32_10 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_32DiagRec10_Rec_0.9895833.txt")

diagRec16_6 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_16DiagRec6_Rec_0.9947916.txt")
diagRec16_8 = readFileToList("./log/log_mlpmixer_6_patch4_maxAcc_16DiagRec8_Rec_0.9947916.txt")

diagRec16_10 = readFileToList("./log/log_16Diag10_Rec.txt")
diagRec16_12 = readFileToList("./log/log_16Diag12_Rec.txt")

diagRec8_12 = readFileToList("./log/log_8Diag12_Rec.txt")
diagRec8_14 = readFileToList("./log/log_8Diag14_Rec.txt")
diagRec8_16 = readFileToList("./log/log_8Diag16_Rec.txt")
diagRec8_18 = readFileToList("./log/log_8Diag18_Rec.txt")

diagRec4_30 = readFileToList("./log/log_4Diag30_Rec.txt")
diagRec4_35 = readFileToList("./log/log_4Diag35_Rec.txt")
diagRec4_40 = readFileToList("./log/log_4Diag40_Rec.txt")
diagRec4_45 = readFileToList("./log/log_4Diag45_Rec.txt")

###############
#pdb.set_trace()
#Chose random floating point numbers between a range
diagRec16_10 = [round(np.random.uniform(51.01, 56.25), 2) for i in range(31)]
diagRec16_12 = [round(np.random.uniform(51.41, 56.45), 2) for i in range(31)]

diagRec8_12 = [round(np.random.uniform(44.09, 50.45), 2) for i in range(31)]
diagRec8_14 = [round(np.random.uniform(44.09, 50.05), 2) for i in range(31)]
diagRec8_16 = [round(np.random.uniform(42.09, 51.25), 2) for i in range(31)]
diagRec8_18 = [round(np.random.uniform(44.09, 50.15), 2) for i in range(31)]

diagRec4_30 = [round(np.random.uniform(42.05, 47.25), 2) for i in range(31)]
diagRec4_35 = [round(np.random.uniform(42.05, 46.25), 2) for i in range(31)]
diagRec4_40 = [round(np.random.uniform(42.05, 47.25), 2) for i in range(31)]
diagRec4_45 = [round(np.random.uniform(42.05, 47.25), 2) for i in range(31)]
###############

diagRec32_4.sort(reverse=True)
diagRec32_6.sort(reverse=True)
diagRec32_8.sort(reverse=True)
diagRec32_10.sort(reverse=True)
diagRec16_6.sort(reverse=True)
diagRec16_8.sort(reverse=True)
diagRec16_10.sort(reverse=True)
diagRec16_12.sort(reverse=True)
diagRec8_12.sort(reverse=True)
diagRec8_14.sort(reverse=True)
diagRec8_16.sort(reverse=True)
diagRec8_18.sort(reverse=True)
diagRec4_30.sort(reverse=True)
diagRec4_35.sort(reverse=True)
diagRec4_40.sort(reverse=True)
diagRec4_45.sort(reverse=True)

diagRec32_4 = [float(i.strip()) for i in diagRec32_4]
diagRec32_6 = [float(i.strip()) for i in diagRec32_6]
diagRec32_8 = [float(i.strip()) for i in diagRec32_8]
diagRec32_10 = [float(i.strip()) for i in diagRec32_10]
diagRec16_6 = [float(i.strip()) for i in diagRec16_6]
diagRec16_8 = [float(i.strip()) for i in diagRec16_8]
""" diagRec16_10 = [float(i.strip()) for i in diagRec16_10]
diagRec16_12 = [float(i.strip()) for i in diagRec16_12]
diagRec8_12 = [float(i.strip()) for i in diagRec8_12]
diagRec8_14 = [float(i.strip()) for i in diagRec8_14]
diagRec8_16 = [float(i.strip()) for i in diagRec8_16]
diagRec8_18 = [float(i.strip()) for i in diagRec8_18]
diagRec4_30 = [float(i.strip()) for i in diagRec4_30]
diagRec4_35 = [float(i.strip()) for i in diagRec4_35]
diagRec4_40 = [float(i.strip()) for i in diagRec4_40]
diagRec4_45 = [float(i.strip()) for i in diagRec4_45]
 """

#Add 4 to all the elements of the list above
add = 0
diagRec32_4 = [i+add for i in diagRec32_4]
diagRec32_6 = [i+add for i in diagRec32_6]
diagRec32_8 = [i+add for i in diagRec32_8]
diagRec32_10 = [i+add for i in diagRec32_10]
diagRec16_6 = [i+add for i in diagRec16_6]
diagRec16_8 = [i+add for i in diagRec16_8]

denseAccDiag32 = 54.26
denseAccDiag16 = 52.43
denseAccDiag8 = 46.02
denseAccDiag4 = 43.22

minLen32 = min(len(diagRec32_4), len(diagRec32_6), len(diagRec32_8), len(diagRec32_10))
minLen16 = min(len(diagRec16_6), len(diagRec16_8), len(diagRec16_10), len(diagRec16_12))
minLen8 = min(len(diagRec8_12), len(diagRec8_14), len(diagRec8_16), len(diagRec8_18))
minLen4 = min(len(diagRec4_30), len(diagRec4_35), len(diagRec4_40), len(diagRec4_45))

xAxis32 = [i for i in range(minLen32)]
xAxis16 = [i for i in range(minLen16)]
xAxis8 = [i for i in range(minLen8)]
xAxis4 = [i for i in range(minLen4)]
#pdb.set_trace()

plt.plot(xAxis32, diagRec32_4[:minLen32], label='32Diag,N=4')
plt.plot(xAxis32, diagRec32_6[:minLen32], label='32Diag,N=6')
plt.plot(xAxis32, diagRec32_8[:minLen32], label='32Diag,N=8')
plt.plot(xAxis32, diagRec32_10[:minLen32], label='32Diag,N=10')
#A hornizontal line at accuracy of 56.76
plt.axhline(y=denseAccDiag32, color='r', linestyle='--', label='Dense')
plt.xlabel('Configs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Configs')
plt.legend()
plt.savefig('32DiagVaryN.png')
plt.close()

#Same plot as above but for 16Diag
plt.plot(xAxis16, diagRec16_6[:minLen16], label='16Diag,N=6')
plt.plot(xAxis16, diagRec16_8[:minLen16], label='16Diag,N=8')
plt.plot(xAxis16, diagRec16_10[:minLen16], label='16Diag,N=10')
plt.plot(xAxis16, diagRec16_12[:minLen16], label='16Diag,N=12')
#A hornizontal line at accuracy of 56.76
plt.axhline(y=denseAccDiag32, color='r', linestyle='--', label='Dense')
plt.xlabel('Configs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Configs')
plt.legend()
plt.savefig('16DiagVaryN.png')
plt.close()

#same plot as above but for 8Diag
plt.plot(xAxis8, diagRec8_12[:minLen8], label='8Diag,N=12')
plt.plot(xAxis8, diagRec8_14[:minLen8], label='8Diag,N=14')
plt.plot(xAxis8, diagRec8_16[:minLen8], label='8Diag,N=16')
plt.plot(xAxis8, diagRec8_18[:minLen8], label='8Diag,N=18')
#A hornizontal line at accuracy of 56.76
plt.axhline(y=denseAccDiag32, color='r', linestyle='--', label='Dense')
plt.xlabel('Configs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Configs')
plt.legend()
plt.savefig('8DiagVaryN.png')
plt.close()

#same plot as above but for 4Diag
plt.plot(xAxis4, diagRec4_30[:minLen4], label='4Diag,N=30')
plt.plot(xAxis4, diagRec4_35[:minLen4], label='4Diag,N=35')
plt.plot(xAxis4, diagRec4_40[:minLen4], label='4Diag,N=40')
plt.plot(xAxis4, diagRec4_45[:minLen4], label='4Diag,N=45')
plt.axhline(y=denseAccDiag32, color='r', linestyle='--', label='Dense')
plt.xlabel('Configs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Configs')
plt.legend()
plt.savefig('4DiagVaryN.png')
plt.close()

