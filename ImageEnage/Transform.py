#第一步：基于HSI颜色变换
import cv2
import numpy as np
#RGB->HSI
from numpy import double


def __Rgb2Hsi(R, G, B):
    # 归一化到[0,1]
    R /= 255
    G /= 255
    B /= 255
    eps =  1e-8
    H, S, I = 0, 0, 0
    sumRGB = R + G + B
    Min = min(R,G,B)
    S = 1 - 3 * Min / (sumRGB + eps)
    H = np.arccos((0.5 * (R + R - G - B)) / np.sqrt((R - G) * (R - G) + (R - B) * (G - B) + eps))
    if B > G:
        H = 2 * np.pi - H
    H = H / (2 * np.pi)
    if S == 0:
        H = 0
    I = sumRGB / 3
    return np.array([H, S, I], dtype = float)

def Rgb2Hsi(img):
    HSIimg = np.zeros(img.shape, dtype = float)
    width, height = img.shape[:2]
    for w in range(width):
        for h in range(height):
            HSIimg[w,h,:] = __Rgb2Hsi(img[w,h,0],img[w,h,1],img[w,h,2])
    return HSIimg


#HSI->RGB
def __Hsi2Rgb(H, S, I):
    pi3 = np.pi / 3
    # 扩充弧度范围[0,2pi]
    H *= 2 * np.pi
    if H >= 0 and H < 2 * pi3:
        # [0,2pi/3)对应红->绿
        B = I * (1 - S)
        R = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        G = 3 * I - (R + B)
    elif H >= 2 * pi3 and H <= 4 * pi3:
        # [2pi/3,4pi/3)对应绿->蓝
        H = H - 2 * pi3
        R = I * (1 - S)
        G = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        B = 3 * I - (R + G)
    else:
        # [4pi/3,2pi)对应蓝->红
        H = H - 4 * pi3
        G = I * (1 - S)
        B = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        R = 3 * I - (B + G)
    return (np.array([R,G,B]) * 255).astype(np.uint8)

def Hsi2Rgb(img):
    RGBimg = np.zeros(img.shape, dtype = np.uint8)
    width, height = img.shape[:2]
    for w in range(width):
        for h in range(height):
            RGBimg[w,h,:] = __Hsi2Rgb(img[w,h,0],img[w,h,1],img[w,h,2])
    return RGBimg


def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j] - g[i, j]) + (r[i, j] - b[i, j]))
            den = np.sqrt((r[i, j] - g[i, j]) ** 2 + (r[i, j] - b[i, j]) * (g[i, j] - b[i, j]))
            theta = float(np.arccos(num / (den*1.0)))
            if den == 0:
                H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j] + g[i, j] + r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * 3.14159265)
            I = sum / 3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H * 255
            hsi_lwpImg[i, j, 1] = S * 255
            hsi_lwpImg[i, j, 2] = I * 255
    return hsi_lwpImg


if __name__ == '__main__':
    rgb_lwpImg = cv2.imread("123.jpg")
    hsi_lwpImg = rgbtohsi(rgb_lwpImg)

    cv2.imshow('rgb_lwpImg', rgb_lwpImg)
    cv2.imshow('hsi_lwpImg', hsi_lwpImg)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()