import cv2
from PIL import Image
from matplotlib import pyplot as plt

from ImageEnage.ACE import zmIceColor
from ImageEnage.GB import GaussianBlur
from ImageEnage.One import _pm
from ImageEnage.Transform import Hsi2Rgb, rgbtohsi


def got_RGB(img_path):
    img = Image.open(img_path)
    width, height = img.size
    img = img.convert('RGB')
    array = []
    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((i, j))
            if r != 0:
                print(r, b, g)
            rgb = (r, g, b)
            array.append(rgb)

#图像路径
image_path='Nan.jpg'

# #将原图像转化为RGB颜色通道的数据
# path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Nan.jpg")
# img = Image.open(path_img)
# width, height = img.size
# img1 = Image.open(path_img).convert('RGB')
# #获取RGB通道的三个分量R、G、B
# for i in range(width):
#     for j in range(height):
#         r, g, b = img.getpixel((i, j))
#
# #RGB->HSI通道的三个分量H、S、I
# HSI_Img=__Rgb2Hsi(r,g,b)
# #转化为HSI通道的图像信息

Img_Hsi= rgbtohsi(cv2.imread("out.jpg"))

#图像亮度自适应调整
#img = aug(Img_Hsi)
m = zmIceColor(cv2.imread('Test.jpg')/255.0)*255
img=_pm(m)
#HSI——>RGB
RGB_new=Hsi2Rgb(img)



#高斯模糊图像处理

#img2 = GaussianBlur(sigma=4.5).filter(Image.open("out.jpg").convert("RGB"))
img2 = GaussianBlur(sigma=4.5).filter(RGB_new)
plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(img2)

# dpi参数维持图片的清晰度
plt.savefig("gaussian.jpg", dpi=4000)
plt.show()
