
#多尺度MSR(Multi-Scale Retinex)
# MSR也可实现色彩增强、颜色恒常性、局部动态范围压缩、全局动态范围压缩，也可以用于X光图像增强
import time

import numpy as np
import cv2


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


if __name__ == '__main__':
    begin=time.time()
    # ../Image/Forest3.jpg
    img = '../Image/foggy_bench.jpg'
    scales = [15,101,301]  # [3,5,9]  #看不出效果有什么差别
    src_img = cv2.imread(img)
    b_gray, g_gray, r_gray = cv2.split(src_img)
    b_gray = MSR(b_gray, scales)
    g_gray = MSR(g_gray, scales)
    r_gray = MSR(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])

    cv2.imwrite('MSR_foggy_bench.jpg',result)
    end=time.time()
    print(int(round(end * 1000)) - int(round(begin * 1000)))