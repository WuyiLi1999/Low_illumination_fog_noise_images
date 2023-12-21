import numpy as np
from PIL import Image

def calculate_rmse(original_image_path, enhanced_image_path):
    # 读取原始图像和增强后的图像
    original_image = Image.open(original_image_path)
    enhanced_image = Image.open(enhanced_image_path)

    # 转换为灰度图像
    original_image_gray = original_image.convert('L')
    enhanced_image_gray = enhanced_image.convert('L')

    # 将图像转换为NumPy数组
    original_array = np.array(original_image_gray)
    enhanced_array = np.array(enhanced_image_gray)

    # 计算差值的平方，并对所有像素求和
    mse = np.mean((original_array - enhanced_array) ** 2)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    return rmse

originalPath = "../resultImages/bench/bench.jpg"
enhancePath = "../resultImages/bench/this_beach.jpg"

result = calculate_rmse(originalPath,enhancePath)
print(result)