"""
Heat Equation
"""

# Author: Khoa Nguyen <khoa.v18nguyen@gmail.com>

import copy
import scipy as sc
from scipy import ndimage

def _he(u, N=100, alpha=1, lambda_=0.05):
    """ Heat equation diffusion
    :param lambda_: float
            learning rate
    :param u: ndarray of the image's shape
            observed image
    :param N: int
            number of iterations
    :param alpha: float
            diffusion time step
    :return:
        uf: ndarray of the image's shape
            de-noised image
    """

    uf = copy.deepcopy(u)  # initialization

    for k in range(N):
        du_dt = 2 * lambda_ * ((u - uf) + alpha * sc.ndimage.filters.laplace(uf))
        uf = uf + du_dt

    return uf