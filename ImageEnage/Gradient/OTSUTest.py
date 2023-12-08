#coding:utf-8
import cv2
from matplotlib import pyplot as plt

image = cv2.imread("../PM/ACE.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(131), plt.imshow(gray, "gray")
plt.title("A:Gray image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.hist(image.ravel(), 256,rwidth=0.8,range=(0,255))
plt.title("B:Histogram")
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
# 对原始图像进行分割
foreground = cv2.bitwise_and(image, image, mask=ret1)
background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(ret1))

plt.subplot(133),plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.subplot(134),plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

plt.subplot(135), plt.imshow(th1, "gray")
plt.title("C:OTSU,threshold K is " + str(ret1)),plt.xticks([]), plt.yticks([])
plt.show()