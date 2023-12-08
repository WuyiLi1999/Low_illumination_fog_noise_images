# rgb均值及标准差
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
img = cv2.imread("/Image/waterfall.jpg")
[height_src0, width_src0,hhh]= img.shape
num = height_src0 * width_src0  # 这里宽高是每幅图片的大小
R_channel = np.sum(img[:, :, 0])
G_channel = np.sum(img[:, :, 1])
B_channel = np.sum(img[:, :, 2])

R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

mean=(R_mean+G_mean+B_mean)/3
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
#标准差
biaozhuncha=img.std()
#RGB均值，和上面的求出的结果是一样的
print(img[:, :, 0].mean())# r
print(img[:, :, 1].mean())# g
print(img[:, :, 2].mean())# b
print("均值：",mean)
print("标准差：",biaozhuncha)

