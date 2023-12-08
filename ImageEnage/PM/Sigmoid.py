import math

import cv2


def sigmoid_function(p):
    x,y,z=p.shape
    fz = []
    for i in range(len(p)):
        for j in range(len(img[i])):
            for k in range(z):
                p[i][j][k]=p[i][j][k]/(1+math.exp(- p[i][j][k]))
                fz.append(1/(1 + math.exp(-num)))
    return p

if __name__ == '__main__':
    img=cv2.imread('TVtest.jpg')
    img=sigmoid_function(img)
    img2=cv2.imread('py_recover_waterfall.jpg')
    Contrastimg = cv2.addWeighted(img, 1.5, img2, 2, 0)  # 调整对比度
    brightness = cv2.addWeighted(img, 1, img2, 2, 40)  # 调整亮度
    cv2.imwrite('SimoidTest.jpg',img)
