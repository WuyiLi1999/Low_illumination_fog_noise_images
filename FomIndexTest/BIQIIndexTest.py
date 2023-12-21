import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_biqi(original_image_path, enhanced_image_path):
    # 读取原始图像和增强后的图像
    original_image = Image.open(original_image_path)
    enhanced_image = Image.open(enhanced_image_path)

    # 将图像转换为NumPy数组
    original_array = np.array(original_image)
    enhanced_array = np.array(enhanced_image)

    # 计算结构相似度（SSIM）
    ssim_score, _ = ssim(original_array, enhanced_array, multichannel=True, full=True)

    # 计算结构失真度（D）
    distortion = 1 - ssim_score

    # 计算BIQI
    biqi = 1 - (1 / (1 + distortion))

    return biqi

originalPath = "../resultImages/Tian'anmen/Door.jpg"
enhancePath = "../resultImages/Tian'anmen/this_door.jpg"

result = calculate_biqi(originalPath,enhancePath)
print(result)