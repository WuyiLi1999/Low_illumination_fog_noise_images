#第三步：对RGB颜色空间的图像进行高斯模糊（平滑）处理，保留图像细节
#高斯模糊(高斯平滑进行二维卷积)与颜色重构

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class GaussianBlur(object):
    def __init__(self, kernel_size=3, sigma=1.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()

    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=np.float)
        radius = self.kernel_size // 2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                # 二维高斯函数
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
        kernel2 = kernel / np.sum(kernel)
        return kernel2
    #img: Image.Image
    def filter(self, img: Image.Image):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            new_arr = signal.convolve2d(img_arr, self.kernel, mode="same", boundary="symm")
        else:
            h, w, c = img_arr.shape
            new_arr = np.zeros(shape=(h, w, c), dtype=np.float)
            for i in range(c):
                new_arr[..., i] = signal.convolve2d(img_arr[..., i], self.kernel, mode="same", boundary="symm")
        new_arr = np.array(new_arr, dtype=np.uint8)
        return Image.fromarray(new_arr)


def main():
    img = Image.open("foggy_bench.jpg").convert("RGB")
    img2 = GaussianBlur(sigma=2.5).filter(img)

    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)

    # dpi参数维持图片的清晰度
    plt.savefig("GB_foggy_bench.jpg", dpi=500)
    plt.show()
