#计算图像亮度
import math

import cv2
import numpy as np
from PIL import ImageStat
from PIL import Image

#计算灰度图像的平均亮度值
def brightness_Ave( im_file ):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]
#计算灰度图像的RMS像素亮度
def brightness_RMS( im_file ):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.rms[0]
#平均像素->可感知亮度
def brightness_1( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
#像素的均方根，然后转换为“感知亮度”
def brightness_2( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   r,g,b = stat.rms
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

#计算像素的“感知亮度”，然后返回平均值
def brightness_3( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
         for r,g,b in im.getdata())
   return sum(gs)/stat.count[0]


def getImageVar(imgPath):
   image = cv2.imread(imgPath)
   img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
   return imageVar

if __name__ == '__main__':
   #origin = cv2.imread('waterfall.jpg')
   # gray = cv2.cvtColor(origin,cv2.COLOR_BGR2GRAY)
   #cv2.imwrite('gray.jpg',gray)
   #brigntness=brightness_3('/Image/waterfall.jpg');
   print('平均亮度：',getImageVar('/Image/waterfall.jpg'))