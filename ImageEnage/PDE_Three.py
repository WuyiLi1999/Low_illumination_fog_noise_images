from PIL import Image
from numpy import *
import numpy as  np

# Space and time domain
M,  N,  K   = 540,  540,  4000
Dx, Dy, Dt  = 0.03, 0.03, 1e-4

# Material Properties
tau = 3e-4
eps_bar = 0.01
sigma = 0.02
J = 4.
theta_0 = 0.2
alpha = 0.9
gamma = 10.
T_eq = 1.
kappa = 1.8

#%% Evolution
p = zeros((M, N))
T = zeros((M, N))
# Initial Solidification area
for i in range(M):
    for j in range(N):
        if (i - M/2)**2 + (j - N/2)**2 < 5.0:
            p[i, j] = 1.0
img = Image.open("foggy_bench.jpg").convert("RGB")
p=np.ndarray(img)
# Define Laplacian operator
def Lap(p):
    p_i_j  = delete(delete(p, [0, -1], axis=0), [0, -1], axis=1)
    p_im_j = delete(delete(p, [0, -1], axis=0), [-1,-2], axis=1)
    p_ip_j = delete(delete(p, [0, -1], axis=0), [0,  1], axis=1)
    p_i_jm = delete(delete(p, [0, -1], axis=1), [0,  1], axis=0)
    p_i_jp = delete(delete(p, [0, -1], axis=1), [-1,-2], axis=0)
    Lap_p  = (p_im_j + p_ip_j + p_i_jm + p_i_jp - 4*p_i_j)/Dx**2
    Lap_pj = vstack((Lap_p[0,:], Lap_p, Lap_p[-1,:]))
    return hstack((Lap_pj[:,0].reshape(N,1), Lap_pj, Lap_pj[:,-1].reshape(N,1)))

# Phase field evolution
def Phase_field(p, T):
    theta = arctan2(gradient(p, Dy, axis=1), gradient(p, Dx, axis=0))
    eps = eps_bar * (1. + sigma * cos(J * (theta - theta_0)))
    g = -eps * eps_bar * sigma * J * sin(J * (theta - theta_0)) * gradient(p, Dy, axis=1)
    h = -eps * eps_bar * sigma * J * sin(J * (theta - theta_0)) * gradient(p, Dx, axis=0)
    m = alpha/pi * arctan(gamma * (T_eq - T))
    term_1 = - p*(p - 1.0)*(p - 0.5 + m)
    term_2 = - gradient(g, Dx, axis=0)
    term_3 = gradient(h, Dy, axis=1)
    term_4 = eps**2 * Lap(p)
    p_ev = Dt / tau * (term_1 + term_2 + term_3 + term_4)
    return p + p_ev

# Temperature evolution
def Temp(T, p_new, p_old):
    T_ev = Dt*Lap(T) + kappa*(p_new - p_old)
    return T + T_ev

# Evolution process
p_hist = []
T_hist = []
p_old = p; T_old = T
for t_step in range(K):
    p_new = Phase_field(p_old, T_old)
    T_new = Temp(T_old, p_new, p_old)
    p_old = p_new
    T_old = T_new
    if t_step % 50 == 0:
        p_hist.append(p_new)
        T_hist.append(T_new)
        print('step finished:', t_step,'/',str(K))