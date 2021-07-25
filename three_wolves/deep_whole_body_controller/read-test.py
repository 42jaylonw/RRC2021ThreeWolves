import numpy as np
file_name = ['0', '1', '2', '3', '4', '5']
arr = []
for f in file_name:
    arr.append(np.load(f + '.npy'))
    print(np.round(np.load(f + '.npy'), 5))

# for i in np.arange(4):
#     print(arr[i], '\n')