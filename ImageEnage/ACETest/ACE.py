#第二步：局部自适应滤波重建亮度--并最后转化为RGB色彩空间
#自动色彩均衡算法ACE
#Rizzi等依据Retinex理论提出了自动颜色均衡算法，该算法考虑了图像中颜色和亮度的空间位置关系，进行局部特性的自适应滤波，
# 实现具有局部和非线性特征的图像亮度与色彩调整和对比度调整，同时满足灰色世界理论假设和白色斑点
#OpenCV—python 自动色彩均衡（ACE）
import os
import time

import cv2
import numpy as np
import math

from PIL import Image

from ImageEnage.Transform import __Rgb2Hsi, rgbtohsi


def stretchImage(data, s=0.005, bins = 2000): #线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0])/float(data.size)
    lmin = 0; lmax=bins-1
    while lmin<bins:
        if d[lmin]>=s:
            break
        lmin+=1
    while lmax>=0:
        if d[lmax]<=1-s:
            break
        lmax-=1
    return np.clip((data-ht[1][lmin])/(ht[1][lmax]-ht[1][lmin]), 0,1)

g_para = {}

def getPara(radius = 5): #根据半径计算权重参数矩阵
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius*2+1
    m = np.zeros((size, size))
    for h in range(-radius, radius+1):
        for w in range(-radius, radius+1):
            if h==0 and w==0:
                continue
            m[radius+h, radius+w] = 1.0/math.sqrt(h**2+w**2)
            m /= m.sum()
            g_para[radius] = m
    return m

def zmIce(I, ratio=4, radius=300): #常规的ACE实现
    para = getPara(radius)
    height,width = I.shape
    zh,zw = [0]*radius + [x for x in range(height)] + [height-1]*radius, [0]*radius + [x for x in range(width)] + [width -1]*radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius*2+1):
        for w in range(radius*2+1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I-Z[h:h+height, w:w+width])*ratio, -1, 1))
    return res


def zmIceFast(I, ratio, radius): #单通道ACE快速增强实现
    height, width = I.shape[:2]
    if min(height, width) <=2:
        return np.zeros(I.shape)+0.5
    Rs = cv2.resize(I, ((width+1)//2, (height+1)//2))
    Rf = zmIceFast(Rs, ratio, radius) #递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))
    return Rf+zmIce(I,ratio, radius)-zmIce(Rs,ratio,radius)


def zmIceColor(I, ratio=2, radius=4): #rgb三通道分别增强，ratio是对比度增强因子2，radius是卷积模板半径4
    res = np.zeros(I.shape)
    for k in range(3):
        res[:,:,k] = stretchImage(zmIceFast(I[:,:,k], ratio, radius))
    return res

if __name__ == '__main__':
    #获取原图像对应的图像矩阵--RGB三通道
    #Img_Rgb = cv2.imread("waterfall.jpg")
    #将RGB通道转换为HSV通道
    # HSV_img = cv2.cvtColor(Img_Rgb, cv2.COLOR_BGR2HSV)
    # m1=(zmIceColor(HSV_img/255.0)*255).astype(np.float32)
    # Img_Rgb=cv2.cvtColor(m1,cv2.COLOR_HSV2BGR)
    begin=time.time()
    m = zmIceColor(cv2.imread('waterfall.jpg')/255.0)*255  #cv2.imread('Test.jpg')/255.0
    #m=zmIce(cv2.imread('waterfall.jpg')/255.0)*255
    end=time.time()
    print(int(round(end * 1000)) - int(round(begin * 1000)))
    cv2.imwrite('zmIceOne.jpg', m)
    #cv2.imwrite('TestTwo.jpg',Img_Rgb)