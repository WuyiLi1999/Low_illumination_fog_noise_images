import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_segmentation(image):
    # 将彩色图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算OTSU阈值
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 对原始图像进行分割
    foreground = cv2.bitwise_and(image, image, mask=threshold)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(threshold))

    return gray,foreground, background,_,threshold

# 读取输入图像
image = cv2.imread('../ACETest/ACEDoor.jpg')

# 进行图像分割
gray,foreground, background,_,threshold = otsu_segmentation(image)

# 显示分割结果
plt.subplot(1, 3, 1), plt.imshow(gray, "gray")
plt.axis('off')
# plt.subplot(1, 4, 2), plt.hist(image.ravel(), 256,rwidth=0.8,range=(0,255))
# plt.subplot(1, 3, 2),plt.imshow(threshold,"gray")
# plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.title(str(_))
plt.axis('off')

plt.show()