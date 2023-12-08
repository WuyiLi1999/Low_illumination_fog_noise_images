"""
Perona Malik Diffusion
"""

# Author: Khoa Nguyen <khoa.v18nguyen@gmail.com>

import copy
import numpy as np


def _div(v, K, esf):
    """Divergence operator"""

    grad_x = np.gradient(v)[0]
    grad_y = np.gradient(v)[1]

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


def _pm(u, N=50, lambda_=0.07, K=0.2, esf='c1'):
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