import numpy as np
from PIL import Image
from scipy.ndimage import sobel

def calculate_sdme(original_image_path, enhanced_image_path):
    # 读取原始图像和增强后图像
    original_image = Image.open(original_image_path)
    enhanced_image = Image.open(enhanced_image_path)

    # 将图像转换为灰度图
    original_gray = original_image.convert('L')
    enhanced_gray = enhanced_image.convert('L')

    # 将灰度图转换为NumPy数组
    original_array = np.array(original_gray)
    enhanced_array = np.array(enhanced_gray)

    # 计算原始图像和增强后图像的结构信息（梯度）
    original_gradient = sobel(original_array)
    enhanced_gradient = sobel(enhanced_array)

    # 计算SDME评价指标
    sdme = np.mean(enhanced_gradient - original_gradient)

    return sdme

originalPath = "../resultImages/bench/bench.jpg"
enhancePath = "../resultImages/bench/Ref[22].jpg"

result = calculate_sdme(originalPath,enhancePath)
print(result)