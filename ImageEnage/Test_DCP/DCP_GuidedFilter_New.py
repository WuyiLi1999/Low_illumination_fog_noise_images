#暗通道先验和引导滤波进行图像去雾（在引导滤波的基础上加上一个复原函数并进行归一化处理）
#引导滤波器进行图像平滑
import time

import cv2
import cv2 as cv
import numpy as np
from scipy.ndimage import minimum_filter
from soft_matting import SoftMatting


class Defog:

    def __init__(self, path, window, a_thresh, omega, t_thresh):
        self.im_ori = cv.imread(path)
        self.im = self.im_ori.astype('float64') / 255
        self.M, self.N, _ = self.im.shape
        self.window = window
        self.a_thresh = a_thresh
        self.omega = omega
        self.t_thresh = t_thresh

    def __get_dark_channel(self, im):
        # w = self.window//2
        # im_padding = np.pad(im, [[w, w], [w, w], [0, 0]], mode='reflect')
        # im_dark = np.zeros([self.M, self.N])
        # for i, j in np.ndindex((self.M, self.N)):
        # 	im_dark[i, j] = np.min(im_padding[i:i+w, j:j+w, :])\
        im_dark = np.min(im, axis=2)
        im_dark = minimum_filter(im_dark, [self.window, self.window])

        return im_dark

    # 大气光值的估计
    def __get_atmosphere(self, im_dark):
        im_dark_flat = im_dark.flatten()
        im_flat = np.reshape(self.im, [self.M * self.N, 3])
        # im_gray_flat = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY).flatten()
        t = self.M * self.N // 1000
        indexs = im_dark_flat.argsort()[-t:]
        # index = np.argmax(im_gray_flat.take(indexs))
        # return im_flat[index]
        a = np.mean(im_flat.take(indexs, axis=0), axis=0)
        a = np.minimum(a, self.a_thresh / 255)
        return a

    # 计算雾天图像的估计传输图
    def __get_t(self, a):
        return 1 - self.omega * self.__get_dark_channel(self.im / a)

    def __recovery(self, a, t):
        t_f = np.maximum(t, self.t_thresh)
        t_f = np.reshape(np.repeat(t_f, 3), [self.M, self.N, 3])
        im = np.float64(self.im)
        return (im - a) / t_f + a

    def __guided_filter_t(self, t, r = 120, eps = 0.0001):
        # r: window size
        Gray_img = cv.cvtColor(self.im_ori, cv.COLOR_RGB2GRAY)
        Gray_img = np.float64(Gray_img)/255
        I = cv.boxFilter(Gray_img, -1, (r, r))#引导图像均值
        P = cv.boxFilter(t, -1, (r, r))#原始待滤波图像均值
        corrI=cv.boxFilter(Gray_img*Gray_img, -1, (r, r))#自相关均值
        corrIP=cv.boxFilter(Gray_img * t, -1, (r, r))#互相关均值
        varI = corrI - I*I #自相关协方差
        covIP=corrIP-I*P#互相关协方差

        #计算窗口线性变换参数系数a，b
        a = covIP / (varI + eps)
        b = P - a * I
        #根据公式计算参数a，b的均值
        a_mean = cv.boxFilter(a, -1, (r, r))
        b_mean = cv.boxFilter(b, -1, (r, r))
        #利用参数得到殷大路博得输出图像
        gf_img = a_mean * Gray_img + b_mean
        return gf_img

    def __aug(self, image, min_bound, max_bound):
        min_bound_pixel = np.percentile(image, min_bound)#1%的分位数
        max_bound_pixel = np.percentile(image, max_bound)#99%的分位数

        image[image >= max_bound_pixel] = max_bound_pixel
        image[image <= min_bound_pixel] = min_bound_pixel

        ret = np.zeros(image.shape, image.dtype)
        # 归一化处理 范围归一化的下界0.1 极差归一化时的上极差边界0.99
        cv.normalize(image, ret, 0.1, 0.99, cv.NORM_MINMAX)

        return ret

    def defog_raw(self):
        dark = self.__get_dark_channel(self.im)
        A = self.__get_atmosphere(dark)
        t = self.__get_t(A)
        i_t = self.__recovery(A, t)

        print(A)
        cv.imshow("t", t)
        cv.imshow("defog with t", i_t)
        cv.waitKey(0)
        cv.destroyAllWindows()
    #基于暗通道DCP和引导滤波器
    def defog_gf(self):
        dark = self.__get_dark_channel(self.im)
        #大气光值的估计
        A = self.__get_atmosphere(dark)
        # 计算雾天图像的估计传输图
        t = self.__get_t(A)
        Guided_Filtered_T = self.__guided_filter_t(t)
        #DCP处理后图像进行颜色恢复
        i_t = self.__recovery(A, t)
        # 对引导滤波平滑后图像进行颜色恢复
        i_gf = self.__recovery(A, Guided_Filtered_T)
        #引导滤波归一化处理
        i_gf_2 = self.__aug(i_gf, 1, 99)

        print(A)
        cv.imshow("t", t)
        cv.imshow("gf_t", Guided_Filtered_T)
        cv.imshow("defog with t", i_t)
        cv.imshow("defog with gf_t", i_gf)
        cv.imshow("defog with gf_t_2", i_gf_2)
        cv.waitKey(0)
        cv.destroyAllWindows()
    #基于暗通道的soft matting(软抠图法)
    def defog_soft_matting(self, scale = 9):
        begin = time.time()
        dark = self.__get_dark_channel(self.im)
        A = self.__get_atmosphere(dark)
        t = self.__get_t(A)
        im_resized = cv.resize(self.im, [self.M//scale, self.N//scale])
        t_resized = cv.resize(t, [self.M//scale, self.N//scale])
        #基于DCP的软抠图法
        soft_matting = SoftMatting(im_resized, t_resized, epsilon=0.0001, lamb=0.0001)
        t_sm_ = soft_matting.get_t()
        t_sm = cv.resize(t_sm_, [self.N, self.M], interpolation=cv.INTER_CUBIC)
        #色彩恢复
        i_sm = self.__recovery(A, t_sm)
        end = time.time()
        print(int(round(end * 1000)) - int(round(begin * 1000)))
        #归一化处理
        i_sm_2 = self.__aug(i_sm, 1, 99)
        end=time.time()
        print(int(round(end * 1000)) - int(round(begin * 1000)))

        print(A)
        cv.imshow("original", self.im)
        cv.imshow("t", t)
        cv.imshow("sm_t", t_sm)
        cv.imshow("defog with sm_t", i_sm)
        cv2.imwrite('Guilder_1.jpg', i_sm)
        cv.imshow("defog with sm_t 2", i_sm_2)
        cv2.imwrite('Guilder_2.jpg', i_sm_2)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    #../Image/Forest3.jpg
    engine = Defog("../Image/tiananmen.png", window=15, a_thresh=230, omega=0.95, t_thresh=0.1)
    begin=time.time()
    engine.defog_soft_matting()

    end=time.time()
    # print(int(round(end * 1000)) - int(round(begin * 1000)))
    #engine.defog_gf()#DCP+引导滤波进行图像去雾增强+图像平滑增强细节
