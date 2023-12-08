"""
Perona Malik Diffusion
"""

# Author: Khoa Nguyen <khoa.v18nguyen@gmail.com>

import copy
import math

import cv2 as cv
import numpy as np


def _div(v, K, esf):
    """Divergence operator"""
    retVal=123
    grad_x = np.gradient(v)[0]
    grad_y = np.gradient(v)[1]
    # grad_x = cv.Sobel(v, cv.CV_32F, 1, 0)
    # grad_y = cv.Sobel(v, cv.CV_32F, 0, 1)
    # gradx = cv.convertScaleAbs(grad_x)
    # grady = cv.convertScaleAbs(grad_y)
    # x,y,z=grad_x.shape
    # m=grad_x.min()
    # M=grad_x.max()
    # avg=grad_x.mean()
    # for i in np.arange(x):
    #     for j in np.arange(y):
    #         for k in np.arange(z):
    #             if grad_x[i][j][k]<avg:
    #                 grad_x[i][j][k]=grad_x[i][j][k]*(2+1*math.cos(math.pi*(math.fabs(grad_x[i][j][k])-math.fabs(m))/(retVal-m)))
    #             else:
    #                 grad_x[i][j][k] = grad_x[i][j][k] * (1 + 0.5 * math.cos(math.pi * (math.fabs(grad_x[i][j][k])-retVal)/(math.fabs(M)-retVal)))
    #
    # x, y, z = grad_y.shape
    # m = grad_y.min()
    # M = grad_y.max()
    # avg = grad_y.mean()
    # for i in np.arange(x):
    #     for j in np.arange(y):
    #         for k in np.arange(z):
    #             if grad_y[i][j][k]<avg:
    #                 grad_y[i][j][k] = grad_y[i][j][k] * (2+1*math.cos(math.pi*(math.fabs(grad_y[i][j][k])-m)/(retVal-m)))
    #             else:
    #                 grad_y[i][j][k] = grad_y[i][j][k] * (1 + 0.5 * math.cos(math.pi * (math.fabs(grad_y[i][j][k])-retVal)/(M-retVal)))
                # if grad_y[i][j][k] > 0:
                #     grad_y[i][j][k] = grad_y[i][j][k] * (5 + 3 * math.cos(math.pi * grad_y[i][j][k]))
                # else:
                #     grad_y[i][j][k] = grad_y[i][j][k] * (1.5 + 0.5 * math.cos(math.pi * grad_y[i][j][k]))

    norm2 = grad_x**2 + grad_y**2

    if esf == 'c1':
        factor = 1 - np.exp(-norm2/K**2)
    elif esf == 'c2':
        factor = 1/np.sqrt(1 + norm2/K**2)
    elif esf == 'c3':
        factor = 1/(1 + norm2/K**2)
    else:
        raise ValueError(' Edge Stopping Function Errors !')

    return np.gradient(factor*grad_x)[0] + np.gradient(factor*grad_y)[1]
    # radxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    # return radxy
def _pm(u, N=5, lambda_=0.07, K=0.2, esf='c1'):
    """
    :param u: ndarray
        observed image
    :param N: int
            number of iterations
    :param lambda_: float
            learning rate
    :param K: int
            edge stopping function's constant
    :param esf: string
        edge stopping function:
            + c1
            + c2
            + c3
    :return:
        uf: ndarray
            de-noised image result
    """
    uf = copy.deepcopy(u)  # initialization

    for k in range(N):
        uf = uf + lambda_*_div(uf, K, esf)

    return uf