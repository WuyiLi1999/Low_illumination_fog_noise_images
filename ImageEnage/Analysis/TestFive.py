#计算结构相似性指数SSIM
# coding:utf-8
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
def calc_ssim(img1, img2):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.

    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    # img1 = Image.open(img1_path).convert('L')
    # img2 = Image.open(img2_path).convert('L')
    # img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255,multichannel=True)
    return ssim_score

if __name__ == '__main__':
   oringal=cv2.imread('imagePlus/newImage/Door.jpg')
   finish=cv2.imread('imagePlus/newImage/Door[20].jpg')
   #result=calc_ssim('waterfall.jpg','HistogramEqualize.jpg')
   print(calc_ssim(oringal,finish))