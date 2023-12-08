#梯度计算
import cv2
import numpy as np
import numpy as np

img=cv2.imread('../PM/zmIce.jpg')
Img_array=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print(Img_array)
gradient=np.gradient(Img_array)
print('---------------')
print(gradient)
img_new=gradient+Img_array
print(img_new)
#该函数返回的第一个值就是输入的thresh值，第二个就是处理后的图像
retVal, a_img = cv2.threshold(Img_array, 0, 255, cv2.THRESH_OTSU)
print(retVal)

