import cv2
import cv2 as cv
#Sobel算子
from PIL import Image


def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv.convertScaleAbs(grad_y)
    x,y,z=image.shape
    print(gradx.sum()/x+grady.sum()/y)*1.0/2
    value=gradx.sum()+grady.sum()
    print(value*1.0/(x*y))

if __name__ == '__main__':
    image = cv2.imread('py_recover_waterfall.jpg')
    sobel_demo(image)