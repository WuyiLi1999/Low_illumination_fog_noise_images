#计算图像的信噪比SNR 峰值信噪比PSNR
import math

import cv2
import numpy as np

#计算信噪比SNR
# def psnr(img1, img2):
#    mse = np.mean( (img1/255. - img2/255.) ** 2 )
#    if mse < 1.0e-10:
#       return 100
#    PIXEL_MAX = 1
#    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#计算峰值信噪比PSNR
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
   oringal=cv2.imread("imagePlus/newImage/bench.jpg")
   finish=cv2.imread("imagePlus/newImage/Ref[20].jpg")
   print(psnr(oringal,finish))
