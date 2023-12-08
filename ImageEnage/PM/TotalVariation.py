#全变分
"""
Total Variation
"""

# Author: Khoa Nguyen <khoa.v18nguyen@gmail.com>

import copy
import numpy as np


def _div(v, epsilon):
    """ Divergence operator"""

    grad_x = np.gradient(v)[0]
    grad_y = np.gradient(v)[1]

    norm2 = grad_x**2 + grad_y**2

    factor = 1/(np.sqrt(epsilon + norm2))

    return np.gradient(factor*grad_x)[0] + np.gradient(factor*grad_y)[1]


def _tv(u, N=40, epsilon=0.0008, lambda_=0.01, alpha=2):
    """
    :param u: ndarray of the image's shape
            observed image
    :param N: int
            number of iterations
    :param epsilon: float
            stabilizing numerical calculation constant
    :param lambda_: float
            learning rate
    :param alpha: float
            diffusion time step
    :return:
        uf: ndarray of the image's shape
            de-noised image
    """

    uf = copy.deepcopy(u)  # initialization

    for k in range(N):
        uf = uf + lambda_*((u - uf) + alpha*_div(uf, epsilon))

    return uf