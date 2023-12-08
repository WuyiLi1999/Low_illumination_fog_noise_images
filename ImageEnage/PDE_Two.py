import matplotlib.pyplot as plt
from numpy import *

M, N = 100, 100
a, b = 1, 1
hx, hy = a / M, b / N
p, q = 1 / hx ** 2, 1 / hy ** 2
r = -2 * (p + q)

U = zeros((M - 1, M - 1))
for i in range(M - 1):
    U[i, i] = r
    if i < M - 2: U[i, i + 1] = p
    if i > 0:   U[i, i - 1] = p
V = diag([q] * (M - 1))
Zero_mat = zeros((M - 1, M - 1))

A_blc = empty((N - 1, N - 1), dtype=object)  # 矩阵A的分块形式
for i in range(N - 1):
    for j in range(N - 1):
        if i == j:
            A_blc[i, j] = U
        elif abs(i - j) == 1:
            A_blc[i, j] = V
        else:
            A_blc[i, j] = Zero_mat

A = vstack([hstack(A_i) for A_i in A_blc])  # 组装得到矩阵A

x_i = linspace(0, a, M + 1)
y_i = linspace(0, b, N + 1)
F = vstack([-2 * pi ** 2 * sin(pi * x_i[1:M].reshape((M - 1, 1))) * sin(pi * j) for j in y_i[1:N]])

u = dot(linalg.inv(A), F).reshape(M - 1, N - 1)
u_f = vstack([zeros((1, M + 1)),  # 最后组装边界条件得到全域的解
              hstack([zeros((N - 1, 1)), u, zeros((N - 1, 1))]),
              zeros((1, M + 1))])
plt.imshow(u_f)
plt.show()
print(u_f)