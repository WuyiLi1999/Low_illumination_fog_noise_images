#梯度函数--Sobel算子
import cv2 as cv

import numpy

def laplace_demo(image):
# dst = cv.Laplacian(image, cv.CV_32F)
# lpls = cv.convertScaleAbs(dst)
    kernel = numpy.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("laplace_demo", lpls)

def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)
    cv.imwrite('forest_X.tif', gradx)
    cv.imshow("gradient_y", grady)
    cv.imwrite('forest_Y.tif', grady)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imwrite('forest.tif', gradxy)
    cv.imshow("gradient", gradxy)

src = cv.imread("../ACETest/Forest7ACE.jpg", cv.IMREAD_COLOR)
cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)
cv.imshow("lena", src)

sobel_demo(src)

#laplace_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()