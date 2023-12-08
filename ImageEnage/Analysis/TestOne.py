#计算图像的熵值--灰度图像
import math

import cv2
import numpy as np
tmp = []
for i in range(256):
    tmp.append(0)
val = 0
k = 0
res = 0
image = cv2.imread('imagePlus/newImage/this_door.jpg',0)
img = np.array(image)
for i in range(len(img)):
    for j in range(len(img[i])):
        val = img[i][j]
        tmp[val] = float(tmp[val] + 1)
        k =  float(k + 1)
for i in range(len(tmp)):
    tmp[i] = float(tmp[i] / k)
for i in range(len(tmp)):
    if(tmp[i] == 0):
        res = res
    else:
        res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
print("图像熵值：",res)