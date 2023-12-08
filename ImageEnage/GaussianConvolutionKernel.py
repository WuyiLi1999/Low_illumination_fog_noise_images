#第三步：对RGB颜色空间的图像进行高斯模糊（平滑）处理，保留图像细节

import numpy as np

#Python高斯卷积核代码
def gaussian_kernel(self):
	kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=np.float)
	radius = self.kernel_size//2
	for y in range(-radius, radius + 1):  # [-r, r]
		for x in range(-radius, radius + 1):
			# 二维高斯函数
			v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
			kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
	kernel2 = kernel / np.sum(kernel)
	return kernel2

#自行实现的二维离散卷积的python代码
def my_conv2d(inputs: np.ndarray, kernel: np.ndarray):
    # 计算需要填充的行列数目，这里假定mode为“same”
    # 一般卷积核的hw都是奇数，这里实现方式也是基于奇数尺寸的卷积核
    h, w = inputs.shape
    kernel = kernel[::-1, ...][..., ::-1]  # 卷积的定义，必须旋转180度
    h1, w1 = kernel.shape
    h_pad = (h1 - 1) // 2
    w_pad = (w1 - 1) // 2
    inputs = np.pad(inputs, pad_width=[(h_pad, h_pad), (w_pad, w_pad)], mode="constant", constant_values=0)
    outputs = np.zeros(shape=(h, w))
    for i in range(h):  # 行号
        for j in range(w):  # 列号
            outputs[i, j] = np.sum(np.multiply(inputs[i: i + h1, j: j + w1], kernel))
    return outputs