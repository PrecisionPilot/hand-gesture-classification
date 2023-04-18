import numpy as np

array1 = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
array2 = np.array([[7, 6], [5, 4], [3, 2], [1, 0]])

combined = np.vstack((array1, array2))

print(type(array1[0]))