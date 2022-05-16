import numpy as np

a = [[1, 3, 4, 5, 6, 7, 8, 9]]
if len(a) == 1:
    a.append([])
b = np.array(a)
b = np.delete(a, -1, 0)

print(b)
print(b.shape)