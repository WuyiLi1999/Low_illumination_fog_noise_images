import numpy as np
from PIL import Image

def calculate_ame(original_image_path, enhanced_image_path):
    # 读取原始图像和增强后的图像
    original_image = Image.open(original_image_path)
    enhanced_image = Image.open(enhanced_image_path)

    # 转换为灰度图像
    original_image_gray = original_image.convert('L')
    enhanced_image_gray = enhanced_image.convert('L')

    # 将图像转换为NumPy数组
    original_array = np.array(original_image_gray)
    enhanced_array = np.array(enhanced_image_gray)

    # 计算绝对误差，并对所有像素求和
    abs_error = np.abs(original_array - enhanced_array)
    ame = np.mean(abs_error)

    return ame
originalPath = "../resultImages/Tian'anmen/Door.jpg"
enhancePath = "../resultImages/Tian'anmen/this_door.jpg"

result = calculate_ame(originalPath,enhancePath)
print(result)
