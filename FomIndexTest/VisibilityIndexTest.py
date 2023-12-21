import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from scipy.ndimage.filters import gaussian_filter

def calculate_visibility(image_path, sigma):
    # 读取图像
    image = Image.open(image_path)

    # 将图像转换为灰度图
    image_gray = image.convert('L')

    # 将灰度图转换为NumPy数组
    image_array = np.array(image_gray)

    # 计算图像的梯度
    gradient = sobel(image_array)

    # 应用高斯滤波
    smoothed_gradient = gaussian_filter(gradient, sigma)

    # 计算可见度评价指标
    visibility = 1 - (1 / np.sum(smoothed_gradient)) * np.sum(np.exp(-gradient**2 / (2 * sigma**2)))

    return visibility

imagePath="../resultImages/waterwall/waterfall[22].jpg"
result=calculate_visibility(imagePath,1.25)
print(result)

