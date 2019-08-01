import numpy as np
import sys

filename = sys.argv[1]
a = np.loadtxt(filename, delimiter=',')

#mean0 = np.mean(a, axis=1)
#maxidx = np.argsort(mean0)[3:]
#a = a[maxidx, :]



print('------ show mean(std) ----------')
mean = np.mean(a, axis=0)
std = np.std(a, axis=0)

print('{:0.2f}({:0.2f}) / {:0.2f}({:0.2f}) / {:0.2f}({:0.2f})'.format(mean[0], std[0], mean[1],std[1],mean[2],std[2]))
# print('{:0.1f}/{:0.1f}/{:0.1f}'.format(mean[0],mean[1],mean[2]))


print('---- show best results in all repeats(best mean of the three metric) ----')

mean2 = np.mean(a,axis=1)
idx = np.argmax(mean2)
print('{:0.1f}/{:0.1f}/{:0.1f}'.format(a[idx][0], a[idx][1],a[idx][2]))
