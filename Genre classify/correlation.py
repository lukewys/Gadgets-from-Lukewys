import numpy as np
from matplotlib import pyplot as plt


gate_size=64
corr=[]

data=np.load('bach_g64.npy')
data=data[:2]

for i in range(len(data)):
    for j in range(len(data)):
        for m in range(len(data[i])-gate_size):
            seqa=data[i][m:m+gate_size]
            corr.append(np.correlate(seqa,data[j]))