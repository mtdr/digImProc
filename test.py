import numpy as np

test = np.zeros([3, 5], int)
for j in range(0, test.shape[1], 1):
    for i in range(0, test.shape[0], 1):
        test[i][j] = -1
print(test)
