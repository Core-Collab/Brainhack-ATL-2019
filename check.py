import numpy as np

gt = np.genfromtxt("/media/data/Track_2/New_Labels_For_Track_2.csv", delimiter=',')
gt = gt[1:-1][:, 1]
print(gt)
np.savetxt('gt_labels.csv', gt, delimiter=',', newline='\n')
mine = np.genfromtxt("/nethome/lchen483/Programming/brain/labels.csv", delimiter=',')
print(mine.shape)
print(np.mean(gt[:3300] == mine[:3300]))
print(np.mean(gt[3300:] == mine[3300:]))
