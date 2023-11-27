# 导入numpy库，用于矩阵运算
import numpy as np

sz = 3
cnt = 0
tot = 0
while tot < 1000:
    tot += 1
    A = np.random.random((sz, sz))
    for i in range(sz):

        for j in range(i):
            A[i][j] = A[j][i]
            # print(i, j)
        A[i][i] = 1

    b = np.ones(sz)
    print(A)
    x = np.linalg.solve(A, b)
    print(x)
    # print(A@x)
    if np.any(x < 0):
        cnt += 1

print(cnt/tot)
